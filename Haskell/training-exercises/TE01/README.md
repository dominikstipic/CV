# PUH Training: TE01

Welcome to your first Haskell training. Every training exercise (TE) has a single
function in the form of `texxx`, so for training exercise `1.1.1` there will be a
function `te111`.

## Exercises

The exercises are split into 3 concepts:

* if-then-else and guards
* lists and strings
* tuples and list comprehensions

Each concept contains 3 mandatory exercises and 1 extra exercise. You must solve
the 9 mandatory exercises to become eligible for the level battle. You do not
need to solve the extra exercise, and there are no "points" for solving them,
but they might help you understand the subject a bit more.

As always, ask your TA if you need any help.

## How to

To start solving the problem simply remove the `undefined` from the body of a
function / definition and start writing your solution.

Before going any further, you should try to update your package index and possibly
your cabal (cabal is a build tool for Haskell). To update your package index run
the following command:

```
cabal new-update
```

If that's not working, try

```
cabal update
cabal install cabal-install
```

This will install the latest version of `cabal` which has `new-` commands available.

To test your solutions you can open the repl / ghci by opening the terminal in the
folder of your training, and typing in the following command:

```
cabal new-repl
```

This will download the necessary dependencies (if there are any) and load your
`TrainingExercises` module into the repl where you can try to call your functions
with different arguments.

When you make changes to your code you can type `:r` into the repl to reload your
code.

For this training, we have also included the test suite so that you can quickly
check if your solutions are passing some rudimentary tests.

To run the test suite, simply type following into your console (you will have to
exit the repl first, or open another console window):

```
cabal new-test
```

If this fails due to missing hspec, you will have to install it:

```
cabal new-install hspec
```

This will run the tests and print out any errors you may have. Also make sure to
uncomment tests for extra tasks if you are solving them.

**YOU CAN ADD YOUR OWN TESTS BUT DO NOT MODIFY OR REMOVE ANY PRE-EXISTING TESTS!
THERE WILL BE SANCTIONS FOR SUCH ACTIONS!**

PROTIP: You can generate HTML documentation for the whole assignment using the
following command:

```
cabal new-haddock
```

And opening the resulting file (you will get the location in the console) with your
browser.
