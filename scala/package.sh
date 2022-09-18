#!/bin/bash

# put all deps (except the provided ones) in a fat jar, skip tests
sbt 'set test in assembly := {}' clean assembly
