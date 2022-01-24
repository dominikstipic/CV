-- ========================================================================== --
{- |
# PUH Level 1 Battle

  Welcome to your first PUH Battle!

  Unlike training, you're alone here. But don't fret - if you solved all your
  exercises and have a good understanding of topics covered, you should have
  no problems.

  This battle consists of 3 tasks. **You must successfully solve all of them in
  order to win the battle.** Every task has a single function in the form of
  `l1x`, e.g. for the first task you have to write a function `l11`.

  The output will be tested both automatically and manually, which means you must
  take care of the output formatting (spaces, punctuation etc.)

  **Do not change provided type signatures!** They're here to help you. Use them
  wisely.

  Finally, you are allowed (encouraged, even) to **create helper functions** as
  you see fit. However, every helper function **must** have type signatures
  defined.

  The soft deadline is Saturday, 20 October 2018 at 12pm (noon).
  The hard deadline is Saturday, 27 October 2018 at 12pm (noon).

  If you want two takes at this battle, submit the first one (this one) before the
  soft deadline.

  More on this can be read here:
  https://puh.takelab.fer.hr/PUH/readme/blob/master/markdown/course-organisation.md#submission-deadlines.



  You can use cabal or GHCi.

  To use cabal, run

  >> cabal repl

  which will run repl. To reload, type `:r`.

  If that's not working for you, run

  >> ghci

  and load the Level Battle:

  >> :l LevelBattle


  Although you are not allowed to ask for help in solving the exercises, if you
  have any trouble with infrastructure, running repl, or anything similar,
  **please ask for help on Slack** (but do it in public channels, e.g. `#haskell`).

  Good luck!
-}
-- ========================================================================== --

module LevelBattle where
--
import Data.List
import Data.Char
--

-- * L 1.1
--
-- | Write a function that checks how many days a February has.
-- In Gregorian calendar leap years occur:
--   * on every year that is divisible by 4
--   * except every year that is divisible by 100
--   * unless the year is also divisible by 400.
-- https://en.wikipedia.org/wiki/Leap_year
--
-- For example, February 1997 and February 1900 had 28 days as it was not a leap year.
-- On the other hand, February 2000 had 29 days.
--
-- Example:
-- lb11 2018 = 28
--
-- Hint: you can write some "helper" functions to help you solve this problem.

isSpecialYear :: Int -> Bool
isSpecialYear n
  | (n `mod` 100 == 0) && (n `mod` 400 /= 0) = True
  | otherwise = False

lb11 :: Int -> Int
lb11 n 
  | (n `mod` 4 == 0) = if isSpecialYear n then 28 else 29 
  | otherwise = 28

-- ** L 1.2
--
-- | Write a function that takes a string and checks whether it is a palindrome
-- after ignoring spaces and letter cases. The function returns a boolean.
--
-- For example, this **is** a palindrome:
-- "Never a foot too far even"
--
-- And obviously, this is not a palindrome:
-- "Haskell is awesome"
--
-- Hint: you can write some "helper" functions to help you solve this problem.

process :: String -> String
process s = [toLower c | c <- s, c /= ' ']

lb12 :: String -> Bool
lb12 s = process (reverse s) == process s

-- * L 1.3
--
-- | Write a function that takes an integer n and a list, and returns a list of
-- every nth element.
--
-- Examples:
-- lb13 3 "haskell 2018" = "sl28"
-- lb13 2 [2,3,5,7,11,13,17] = [3,7,13]

lb13 :: Int -> [a] -> [a]
lb13 n list = [a | (i,a) <- zip [1..] list, i `mod` n == 0]



