# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:29:19 2019

"""

import pytest

@pytest.yield_fixture(scope="module")
def resource_setup():
    print("\nconnect to db")
    db = {"Red":1,"Blue":2,"Green":3}
    #def resource_teardown():
    yield db
    print("\ndisconnect")
    #request.addfinalizer(resource_teardown)
    #return db


def test_db(resource_setup):
    for k in resource_setup.keys():
        print("color {0} has id {1}".format(k, resource_setup[k]))

def test_red(resource_setup):
    assert resource_setup["Red"] == 1

def test_blue(resource_setup):
    assert resource_setup["Blue"] != 1