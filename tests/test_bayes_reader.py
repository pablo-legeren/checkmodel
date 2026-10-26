"""bayesian() and the Bayesian model reader, without ArviZ, PyMC or bambi.

The DataTree, InferenceData, PyMC and bambi objects below are minimal
stand-ins with just the attributes checkmodel reads, filled with the rstanarm
draws of tests/data/bayes_*.csv.gz. These tests run on every Python version;
tests/test_bayes.py covers the same reader on real ArviZ objects (3.12+).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from checkmodel import (
    BayesianModel,
    UnsupportedModelError,
    bayesian,
    check_model,
    r2_bayes,
)
from checkmodel._bayes import as_bayesian, looks_bayesian
from tests import bayes_models as bm


# ------------------------------------------------------------- stand-ins
class FakeArray:
    """An xarray DataArray: named dimensions, ``.values`` and ``.transpose()``."""

    def __init__(self, values, dims):
        self.values = np.asarray(values, dtype=float)
        self.dims = tuple(dims)

    def transpose(self, *order):
        named = [d for d in order if d is not Ellipsis]
        perm = [self.dims.index(d) for d in named]
        perm += [i for i in range(len(self.dims)) if i not in perm]
        return FakeArray(self.values.transpose(perm), [self.dims[i] for i in perm])


class FakeDataset:
    """An xarray Dataset: ``data_vars``, ``in`` and ``[]``."""

    def __init__(self, **arrays):
        self.data_vars = dict(arrays)

    def __contains__(self, name):
        return name in self.data_vars

    def __getitem__(self, name):
        return self.data_vars[name]


class FakeNode:
    def __init__(self, dataset):
        self.dataset = dataset

    def to_dataset(self):
        return self.dataset


class FakeTree:
    """An ArviZ 1.x DataTree: ``tree["posterior"]`` is a node; a missing group raises KeyError."""

    def __init__(self, **groups):
        self.groups = groups

    def __getitem__(self, name):
        return FakeNode(self.groups[name])


class FakeInferenceData:
    """An ArviZ 0.x InferenceData: groups are attributes holding datasets."""

    def __init__(self, **groups):
        for name, group in groups.items():
            setattr(self, name, group)


OBS = ("chain", "draw", "obs")


def _draws(name):
    table = bm.draws(name)
    chains = table["chain"].nunique()
    params = {p: table[p].to_numpy().reshape(chains, -1) for p in bm.parameter_names(name)}
    blocks = {
        key: bm._block(table, key, chains)
        for key in ("mu", "mum", "yrep", "ll")
        if any(c.startswith(f"{key}_") for c in table.columns)
    }
    y = np.asarray(bm.reference()["models"][name]["y"], dtype=float)
    return params, blocks, y


def _groups(name, *, predictive=True):
    params, blocks, y = _draws(name)
    posterior = {p: FakeArray(v, ("chain", "draw")) for p, v in params.items()}
    # Stored draw-major: the reader must put chains first itself.
    posterior["mu"] = FakeArray(blocks["mu"].transpose(1, 0, 2), ("draw", "chain", "obs"))
    groups = {
        "posterior": FakeDataset(**posterior),
        "observed_data": FakeDataset(y=FakeArray(y, ("obs",))),
    }
    if predictive and "yrep" in blocks:
        groups["posterior_predictive"] = FakeDataset(y=FakeArray(blocks["yrep"], OBS))
        groups["log_likelihood"] = FakeDataset(y=FakeArray(blocks["ll"], OBS))
    return groups


def _ref(name):
    return bm.reference()["models"][name]


# --------------------------------------------------------------- reading
def test_reads_a_datatree_without_arviz():
    fit = bayesian(FakeTree(**_groups("bayes_lm")))
    _, blocks, y = _draws("bayes_lm")
    assert isinstance(fit, BayesianModel)
    np.testing.assert_array_equal(fit.mu, blocks["mu"])
    np.testing.assert_array_equal(fit.y, y)
    np.testing.assert_array_equal(fit.y_rep, blocks["yrep"])
    np.testing.assert_array_equal(fit.log_lik, blocks["ll"])
    assert fit.sigma.shape == (2, 300)
    assert list(fit.parameters) == ["X.Intercept.", "wt", "hp", "sigma"]
    assert (fit.family, fit.model_class, fit.response) == ("gaussian", "Bayesian", "y")
    assert r2_bayes(fit)["R2"] == pytest.approx(_ref("bayes_lm")["r2"]["median"], rel=1e-6)


def test_reads_inference_data_and_datatrees_alike():
    tree = bayesian(FakeTree(**_groups("bayes_logit")), family="binomial")
    old = bayesian(FakeInferenceData(**_groups("bayes_logit")), family="binomial")
    np.testing.assert_array_equal(tree.mu, old.mu)
    assert tree.sigma is None and old.info.is_bernoulli
    assert looks_bayesian(FakeInferenceData(**_groups("bayes_logit")))


def test_marginal_expected_values_and_vector_parameters():
    groups = _groups("bayes_mixed")
    _, blocks, _ = _draws("bayes_mixed")
    post = groups["posterior"].data_vars
    post["mu_marginal"] = FakeArray(blocks["mum"], OBS)
    post["b"] = FakeArray(np.ones((2, 300, 3)), ("chain", "draw", "group"))
    fit = bayesian(FakeTree(**groups), mu_marginal="mu_marginal")
    assert fit.is_mixed and fit.info.is_mixed
    np.testing.assert_array_equal(fit.mu_marginal, blocks["mum"])
    assert {"b[0]", "b[1]", "b[2]"} <= set(fit.parameters)
    assert "mu_marginal" not in fit.parameters
    assert fit.y_rep is None and fit.log_lik is None


def test_reader_errors():
    groups = _groups("bayes_lm")
    with pytest.raises(UnsupportedModelError, match="`posterior` and `observed_data`"):
        bayesian(FakeTree(posterior=groups["posterior"]))
    with pytest.raises(UnsupportedModelError, match=r"pm\.Deterministic\('eta'"):
        bayesian(FakeTree(**groups), mu="eta")
    obs = groups["observed_data"].data_vars
    obs["w"] = obs["y"]
    with pytest.raises(ValueError, match="pass `response=`"):
        bayesian(FakeTree(**groups))
    assert bayesian(FakeTree(**groups), response="y").response == "y"


def test_looks_bayesian_and_as_bayesian():
    fit = bayesian(FakeTree(**_groups("bayes_lm")))
    assert looks_bayesian(fit) and as_bayesian(fit) is fit
    assert isinstance(as_bayesian(FakeTree(**_groups("bayes_lm"))), BayesianModel)
    assert as_bayesian(object()) is None
    assert not looks_bayesian(FakeTree(posterior=FakeDataset()))

    class Broken:
        def __getitem__(self, name):
            raise RuntimeError("not a container")

    assert not looks_bayesian(Broken())


def test_model_type_of_other_families():
    fit = bayesian(FakeTree(**_groups("bayes_lm")), family="poisson")
    assert (fit.info.family, fit.info.link, fit.info.backend) == ("poisson", "log", "bayesian")
    fit.family = "gamma"
    assert fit.info.link == "unknown"
    assert fit.chains == 2 and fit.n_draws == 600 and fit.n_obs == 32


def test_check_model_on_a_fake_tree():
    report = check_model(FakeTree(**_groups("bayes_lm")))
    assert list(report) == ["predictions", "convergence"]


# ------------------------------------------------------------------ PyMC
def _pymc_model(op_name, module="pymc.model.core"):
    """A pymc.Model stand-in whose observed variable comes from an `op_name` op."""
    op = type(op_name, (), {})()
    variable = types.SimpleNamespace(owner=types.SimpleNamespace(op=op))
    cls = type("Model", (), {"__getitem__": lambda self, name: variable})
    cls.__module__ = module
    return cls()


@pytest.mark.parametrize(
    ("op", "family"),
    [("NormalRV", "gaussian"), ("BernoulliRV", "binomial"), ("PoissonRV", "poisson")],
)
def test_family_comes_from_the_pymc_model(op, family):
    fit = bayesian(FakeTree(**_groups("bayes_lm")), _pymc_model(op, module="mylib"))
    assert fit.family == family
    assert fit.model_class == "Bayesian"


def test_unknown_pymc_family_falls_back_to_gaussian():
    assert bayesian(FakeTree(**_groups("bayes_lm")), _pymc_model("GammaRV", "x")).family == (
        "gaussian"
    )
    unreadable = type("Model", (), {})()  # no __getitem__: the family cannot be read
    assert bayesian(FakeTree(**_groups("bayes_lm")), unreadable).family == "gaussian"


@pytest.mark.parametrize("pymc6", [True, False])
def test_pymc_fills_in_predictions_and_log_likelihood(monkeypatch, pymc6):
    _, blocks, _ = _draws("bayes_lm")
    yrep = FakeDataset(y=FakeArray(blocks["yrep"], OBS))
    loglik = FakeDataset(y=FakeArray(blocks["ll"], OBS))
    calls = []

    def sample_posterior_predictive(idata, model, progressbar):
        calls.append("predictive")
        return FakeTree(posterior_predictive=yrep)

    def compute_log_likelihood(idata, model, extend_inferencedata, progressbar):
        calls.append("loglik")
        # PyMC 6 returns the log-likelihood Dataset itself; older versions a tree.
        return loglik if pymc6 else FakeTree(log_likelihood=loglik)

    fake_pm = types.SimpleNamespace(
        sample_posterior_predictive=sample_posterior_predictive,
        compute_log_likelihood=compute_log_likelihood,
    )
    monkeypatch.setitem(sys.modules, "pymc", fake_pm)
    model = _pymc_model("NormalRV")
    fit = bayesian(FakeTree(**_groups("bayes_lm", predictive=False)), model)
    assert calls == ["predictive", "loglik"]
    assert fit.model_class == "PyMC"
    np.testing.assert_array_equal(fit.y_rep, blocks["yrep"])
    np.testing.assert_array_equal(fit.log_lik, blocks["ll"])

    calls.clear()
    bayesian(FakeTree(**_groups("bayes_lm")), model)  # nothing missing: nothing computed
    assert calls == []


# ----------------------------------------------------------------- bambi
class FakeBambi:
    """A bambi.Model stand-in: response, family and the predict/log-likelihood methods."""

    def __init__(self, name, family, *, new_api=True):
        _, self.blocks, _ = _draws(name)
        # bayes_mixed has no replicated data; any draws of the right shape will do here.
        self.blocks.setdefault("yrep", self.blocks["mu"])
        self.blocks.setdefault("ll", np.zeros_like(self.blocks["mu"]))
        term = types.SimpleNamespace(name="y")
        if new_api:
            self.response_term = term
        else:
            self.response_component = types.SimpleNamespace(term=term)
        self.family = types.SimpleNamespace(
            name=family, likelihood=types.SimpleNamespace(parent="mu")
        )
        self.calls = []

    def predict(self, idata, kind, inplace, include_group_specific=True):
        assert inplace is False
        self.calls.append((kind, include_group_specific))
        if kind == "response":
            return FakeTree(
                posterior_predictive=FakeDataset(y=FakeArray(self.blocks["yrep"], OBS))
            )
        mu = self.blocks["mu"] if include_group_specific else self.blocks["mum"]
        return FakeTree(posterior=FakeDataset(mu=FakeArray(mu, OBS)))

    def compute_log_likelihood(self, idata, inplace):
        self.calls.append(("log_likelihood", None))
        return FakeTree(log_likelihood=FakeDataset(y=FakeArray(self.blocks["ll"], OBS)))


FakeBambi.__module__ = "bambi.models"


def _bambi_tree(name, extra=None):
    groups = _groups(name, predictive=False)
    post = groups["posterior"].data_vars
    del post["mu"]  # bambi keeps the expected values out of the posterior
    post.update(extra or {})
    return FakeTree(**groups)


def test_bambi_model_is_read_automatically():
    model = FakeBambi("bayes_lm", "gaussian")
    fit = bayesian(_bambi_tree("bayes_lm"), model)
    assert model.calls == [("response_params", True), ("response", True), ("log_likelihood", None)]
    assert (fit.model_class, fit.family) == ("bambi", "gaussian")
    np.testing.assert_array_equal(fit.mu, model.blocks["mu"])
    assert fit.sigma.shape == (2, 300) and fit.mu_marginal is None
    assert r2_bayes(fit)["R2"] == pytest.approx(_ref("bayes_lm")["r2"]["median"], rel=1e-6)


def test_bambi_multilevel_model_gets_marginal_expected_values():
    model = FakeBambi("bayes_mixed", "gaussian", new_api=False)
    group = {"1|g": FakeArray(np.zeros((2, 300, 4)), ("chain", "draw", "g"))}
    fit = bayesian(_bambi_tree("bayes_mixed", group), model)
    assert ("response_params", False) in model.calls
    np.testing.assert_array_equal(fit.mu_marginal, model.blocks["mum"])
    assert fit.is_mixed


def test_bambi_predictions_already_in_the_tree_are_used():
    model = FakeBambi("bayes_lm", "gaussian")
    tree = _bambi_tree("bayes_lm")
    tree.groups["posterior_predictive"] = FakeDataset(y=FakeArray(model.blocks["yrep"], OBS))
    tree.groups["log_likelihood"] = FakeDataset(y=FakeArray(model.blocks["ll"], OBS))
    fit = bayesian(tree, model)
    assert model.calls == [("response_params", True)]
    np.testing.assert_array_equal(fit.log_lik, model.blocks["ll"])


def test_bambi_bernoulli_is_binomial():
    fit = bayesian(_bambi_tree("bayes_logit"), FakeBambi("bayes_logit", "bernoulli"))
    assert fit.family == "binomial" and fit.sigma is None
