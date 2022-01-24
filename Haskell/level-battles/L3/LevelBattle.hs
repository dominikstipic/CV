-- ========================================================================== --
{- |
= PUH Level 3 Battle

  Welcome to your third PUH Battle.

  This battle consists of 3 tasks. You must successfully solve all of them
  correctly in order to win the battle. Every task has a single function in the
  form of `lb3x`, e.g. for the first task you have to write a function `lb31`.

  The output will be tested both automatically and manually, which means you must
  take care of the output formatting (spaces, punctuation etc.)

  Do not change provided type signatures! They're here to help you. Use them
  wisely.

  This time, you must not use helper functions. Use lambda expressions instead
  and don't rename the provided functions.
-}
-- ========================================================================== --
module LevelBattle where
--
-- some useful imports
-- ( check them out on hoogle )
import Data.Char ( chr, ord )
--

{- * CURRIED FORM, PARTIAL APPLICATION & SECTIONS                             -}

{- ** LB 3.1 -}

{- |
  Define a function which checks if the letter of english alphabet is uppercase.
  Make sure you use sections and partial application for this task and eta
  reduce your solution.
-}

lb31 :: Char -> Bool
lb31 = (`elem` ['A'..'Z']) 

{- * HIGHER ORDER FUNCTIONS, MAP, FILTER & LAMBDA EXPRESSIONS                 -}

{- ** LB 3.2 -}

{- |
  Write a function which takes a `String` and a list of `Int`s and then "zips"
  them together (first `Char` with first `Int` and so on, until the end of the
  shorter list). Filter out all the resulting `Int`s which are higher than
  `maxBound :: Char` or lower than `minBound :: Char` and convert resulting list
  of `Int`s back to `String`.

  You can not use `where` or `let`. If you need more complex function use a
  lambda expressions.

  lb32 "\NUL super Duper stringifyer" [1,2,3,4,5,6,7,8,9] ==> "\SOH\"vyuky(M"
  lb32 "super Duper stringifyer" [1,2,3,4,5,6,7,8,9]      ==> "twsiw&K}y"
-}


lb32 :: String -> [Int] -> String
lb32 x = map (chr).filter (\x -> x > ord (minBound :: Char) && x < ord (maxBound :: Char)).map (\(x,y)-> ord x + y).zip x

{- * COMPOSITION                                                              -}

{- ** LB 3.3-}

{- |
  Write a function that takes a list of functions from `Int -> Int` and a single
  `Int` and checks if any function when applied to that `Int` gives a result
  which is greater than 5.

  Make sure you eta reduce the last argument. To achieve this, remember that `$`
  is a "function application" operator. `flip` will also be useful for this
  task.

  Hint: minimal "nice" solution is a one liner and 43 characters long including
  the function name. Without whitespaces and using ugly argument names, it'll
  be down to 35. You don't have to worry about the length of your solution,
  but try to keep it simple.
-}

lb33 :: [Int -> Int] -> Int -> Bool
lb33 f = any (>5).zipWith ($) f.replicate (length f)