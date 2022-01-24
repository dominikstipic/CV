-- =============================================================================== --
{- |
  Welcome to the extra segment of your second Haskell training.

  This one is meant to fully prepare you for your upcoming level battle.
  You do not have to do it, but if you want to practice a bit more before taking
  on the battle, feel free to give it your best.

  As always, ask your TA if you need any help.
-}
-- =============================================================================== --
--
module TrainingExercises where
import Prelude hiding (lookup)
{- * 2 DIY KEY-VALUE STORAGE

   In this exercise, you will have to implement some polymorphic utility functions
   for working on lists of key-value pairs where keys are strings. Think of
   the list as a simple key-value storage.

   When solving the problem, look at the provided type signatures and try to
   figure out how the function should work. This time you don't have any tests
   to help you.

   Remember: in functional programming, objects are immutable. When "adding" an
   element to a list, you're actually creating a new list with that new object.
 -}

-- ** TE 2.1
--
-- | Write a function for getting a pair with a certain key which should return
-- a list with a single element as a result (or no elements if the key doesn't
-- exist):
findItem :: [(String, a)] -> String -> [(String, a)]
findItem l key = [(k,v) | (k,v) <- l, k == key]

-- ** TE 2.2
--
-- | Write a function that checks if a list contains an element with a certain key:
contains :: [(String, a)] -> String -> Bool
contains l key = not $ null $ findItem l key

-- ** TE 2.3
--
-- | Write a function that tries to retrieve a value with a certain key or throws an error if
-- the key doesn’t exist (example of error function usage : error "I’m an error
-- message"):
lookup :: [(String, a)] -> String -> a
lookup l key
  | contains l key = snd $ head $ findItem l key 
  | otherwise = error "The value with provided key doesn't exist"

-- ** TE 2.4
--
-- | Write a function that inserts a new key value pair. If key already exists than do nothing:
insert :: [(String, a)] -> (String, a) -> [(String, a)]
insert l p
  | contains l $ fst p = l
  | otherwise = p : l


-- ** TE 2.5
--
-- | Write a function that removes a key value pair with the certain key:
remove :: [(String, a)] -> String -> [(String, a)]
remove l key = [(k,v) | (k,v) <- l, k /= key]

-- ** TE 2.6
--
-- | Write a function that updates the value of a certain key (if the key doesn’t exist,
-- the function does nothing) :
update :: [(String, a)] -> String -> a -> [(String, a)]
update l key value
  | contains l key = [if fst p == key then (key, value) else p | p <- l]
  | otherwise = l   




