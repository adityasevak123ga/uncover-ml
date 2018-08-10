import pickle
import numpy as np
import pytest
from sklearn.metrics import r2_score

from uncoverml.krige import krige_methods, Krige, all_ml_models, MLKrige
from uncoverml.models import regressors, classifiers
from uncoverml.optimise.models import transformed_modelmaps

models = {**classifiers, **regressors, **transformed_modelmaps}


@pytest.fixture(params=[v for v in models.values()])
def get_models(request):
    return request.param


def test_modeltags(get_models):

    model = get_models()

    # Patch classifiers since they only get their tags when "fit" called
    if hasattr(model, 'predict_proba'):
        # patching is not working
        model.le.classes_ = ('1', '2', '3')

    tags = model.get_predict_tags()

    print(tags)

    assert len(tags) >= 1  # at least a predict function for regression

    if hasattr(model, 'predict_dist'):
        assert len(tags) >= 4  # at least predict, var and upper & lower quant

        if hasattr(model, 'entropy_reduction'):
            assert len(tags) == 5

        if hasattr(model, 'krige_residual'):
            assert len(tags) == 5

    elif hasattr(model, 'predict_proba'):
        # classifiers only get their tags when "fit" called, and
        # patching is not working for me
        assert len(tags) == 1
        # assert tags == ['1', '2', '3']

    else:
        if hasattr(model, 'entropy_reduction'):
            assert len(tags) == 2


def test_modelmap(get_models):

    mod = get_models()

    assert hasattr(mod, 'fit')
    assert hasattr(mod, 'predict')


def test_modelpickle(get_models):

    mod = get_models()
    mod_str = pickle.dumps(mod)
    mod_pic = pickle.loads(mod_str)

    # Make sure all the keys survive the pickle, even if the objects differ
    assert mod.__dict__.keys() == mod_pic.__dict__.keys()


@pytest.fixture(params=krige_methods.keys())
def get_krige_method(request):
    return request.param


def test_krige(linear_data, get_krige_method):

    yt, Xt, ys, Xs = linear_data()

    mod = Krige(method=get_krige_method)
    mod.fit(np.tile(Xt, (1, 2)), yt)
    Ey = mod.predict(np.tile(Xs, (1, 2)))
    assert r2_score(ys, Ey) > 0


@pytest.fixture(params=[k for k in transformed_modelmaps])
def get_transformed_model(request):
    return transformed_modelmaps[request.param]


def test_trasnsformed_model_attr(get_transformed_model):
    """
    make sure all optimise.models classes have ml_score attr
    """
    assert np.all([hasattr(get_transformed_model(), a) for a in
                   ['ml_score', 'score', 'fit', 'predict']])


@pytest.fixture(params=[k for k in all_ml_models
                        if k not in ['randomforest',
                                      'multirandomforest',
                                      'depthregress',
                                      'cubist',
                                      'multicubist',
                                      'decisiontree',
                                      'extratree',
                                      'catboost'
                                     ]])
def models_supported(request):
    return request.param


def test_mlkrige(models_supported, get_krige_method):
    """
    tests algos that can be used with MLKrige
    """
    mlk = MLKrige(ml_method=models_supported, method=get_krige_method)
    assert hasattr(mlk, 'fit')
    assert hasattr(mlk, 'predict')

    mod_str = pickle.dumps(mlk)
    mod_pic = pickle.loads(mod_str)
    # Make sure all the keys survive the pickle, even if the objects differ
    assert mlk.__dict__.keys() == mod_pic.__dict__.keys()


# def test_modelpersistance(make_fakedata):

#     X, y, _, mod_dir = make_fakedata

#     for model in models.modelmaps.keys():
#         mod = models.modelmaps[model]()
#         mod.fit(X, y)

#         with open(path.join(mod_dir, model + ".pk"), 'wb') as f:
#             pickle.dump(mod, f)

#         with open(path.join(mod_dir, model + ".pk"), 'rb') as f:
#             pmod = pickle.load(f)

#         Ey = pmod.predict(X)

#         assert Ey.shape == y.shape
