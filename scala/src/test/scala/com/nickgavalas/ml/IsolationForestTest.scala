package com.nickgavalas.ml

import org.scalatest.FunSuite

class IsolationForestTest extends FunSuite {
  val model = new IsolationForest("/home/nick/ntua/thesis/python/iforest/sampleModels/model.json")

  test("IsolationForest.score") {
    var actual = model.score(Vector(-2.624069700548209205e+00, -2.615518510597409474e+00))
    var expected = 0.4853898991159712
    assert(actual == expected)

    actual = model.score(Vector(-1.689492108687630445e+00, -2.499247162702992853e+00))
    expected = 0.42513509590478726
    assert(actual == expected)
  }
}
