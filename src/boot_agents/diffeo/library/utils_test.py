from numpy.testing.utils import assert_allclose
from boot_agents.diffeo.library.utils import mod1d


def mod_test():
    assert_allclose(mod1d(0), 0)
    assert_allclose(mod1d(1), -1)
    assert_allclose(mod1d(2), 0)
    assert_allclose(mod1d(-1), -1)
    assert_allclose(mod1d(1.1), -0.9)
    assert_allclose(mod1d(0.1), 0.1)
    assert_allclose(mod1d(-0.1), -0.1)
    assert_allclose(mod1d(-1.1), +0.9)

