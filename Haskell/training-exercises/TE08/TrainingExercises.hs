-- =============================================================================== --
{- |
  Welcome to your eighth Haskell training.

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-08`, create a Merge Request,
  and assign it to your TA.

  Keep in mind that **all exposed functions must have type signatures**
  (helper functions that are defined in local definitions don't)!

  Also there aren't any extra tasks - all tasks are mandatory.

  As always, ask your TA if you need any help.
-}
-- =============================================================================== --
--
module TrainingExercises where
--
import Data.Char
import Data.List
import Data.Ord
--

{- * 8.1 Foldr and Foldl  -}

-- ** TE 8.1.1
--
-- | You are given a string of lower case english letters.
-- Write a function that removes all the adjecent duplicate letters from the string using foldr.
-- The removal process is repeated as long as there are letter pairs to be removed.
--
-- -> Example: "ababba"  ==> "ab"
-- (The first removal will result in "abaa". There is another letter pair in that string, so we
--  need to remove that one too and get "ab").
--
-- -> Example: "ababbaa" ==> "aba"
--
-- -> Example: "ababbba" ==> "ababa"
-- (we remove one pair of "bb", the third letter doesn't have a pair to be removed).

te811 :: String -> String
te811 = foldr f []
  where f x [] = [x]
        f x acc@(y : xs) = if (y == x) then xs else x : acc 
        

-- ** TE 8.1.2
--
-- | Define a function (using foldl) which takes string and returns the number of vowels in it.
--
-- -> Example: "Haskell" ==> 2

te812 :: String -> Int
te812 = foldl f 0
  where vowels = "aeiou"
        f n x  = if (toLower x) `elem` vowels then n + 1 else n  

{- * 8.2 Data types  -}

-- ** TE 8.2.1
--
-- | Define a new data type `Day` that can be any day of the week.
-- Make sure to derive Show.

data Day = Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday deriving Show

-- ** TE 8.2.2
--
-- | Define a function that takes a day and checks if we have a regular PUH lecture on that day.
--
-- -> Example: Wednesday ==> False
-- -> Example: Thursday  ==> True

te822 :: Day -> Bool
te822 Thursday = True
te822 _        = False

-- ** TE 8.2.3
--
-- | Let's create a blockchain!
-- First define a record `Transaction` that can be either `Incoming` or `Outgoing`.
-- Both Incoming and Outgoing transactions should have an Int value.
-- Make sure to derive Show.

data Transaction = Incoming Int | Outgoing Int deriving Show

-- ** TE 8.2.4
--
-- | Write a function that takes a list of transactions and calculates the account balance.
-- Hint: You can use either foldr or foldl for this one, but make a wise decision on which one to use!
--
-- -> Example: [Incoming 15, Outgoing 10, Incoming 3] ==> 8

te824 :: [Transaction] -> Int
te824 = foldl f 0
  where f acc (Incoming n) = acc + n
        f acc (Outgoing n) = acc - n

-- ** TE 8.2.5
--
-- | Write a function that takes a list of transactions and checks if it is possible that all
-- of them are executed without the account balance becoming negative at any time.
-- Hint: You can use either foldr or foldl for this one, but make a wise decision on which one to use!
--
-- -> Example: [Incoming 15, Outgoing 10, Incoming 3] ==> True
-- -> Example: [Outgoing 10, Incoming 15, Incoming 3] ==> False

te825 :: [Transaction] -> Bool
te825 = snd . foldl f (0, True)
  where f (_, False) _          = (0, False)
        f (sum, _) (Incoming n) = (sum + n, sum + n > 0) 
        f (sum, _) (Outgoing n) = (sum - n, sum - n > 0)


