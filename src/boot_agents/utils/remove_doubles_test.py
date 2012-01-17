from . import RemoveDoubles, np


def remove_doubles_test_1():
    rd = RemoveDoubles(0.5)
    print rd
    assert not rd.ready()


def remove_doubles_test_2():
    rd = RemoveDoubles(0)
    z1 = np.random.rand(5)
    z2 = np.random.rand(5)
    assert not rd.ready()
    rd.update(z1)
    assert rd.ready()
    rd.update(z1)
    assert not rd.ready()
    rd.update(z2)
    assert rd.ready()


def remove_doubles_test_3():
    rd_0_1 = RemoveDoubles(0.1)
    rd_0_5 = RemoveDoubles(0.5)
    z1 = np.random.rand(5)
    z2 = z1.copy()
    z2[3] = 0 # real fraction 0.2 

    rd_0_1.update(z1)
    rd_0_1.update(z2)
    assert rd_0_1.ready()

    rd_0_5.update(z1)
    rd_0_5.update(z2)
    assert not rd_0_5.ready()
    z2[4] = 0
    z2[2] = 0
    rd_0_5.update(z2)
    assert rd_0_5.ready()
