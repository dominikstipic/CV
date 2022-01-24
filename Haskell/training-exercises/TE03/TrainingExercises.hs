-- =============================================================================== --
{- |
  Welcome to your third Haskell training. Get ready to rumble.
  Where will you let the pattern matching lead you?

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-03`, create a Merge Request,
  and assign it to your TA.

  Keep in mind that **all exposed functions must have type signatures** (helper functions that are defined in local definitions don't)!

  As always, ask your TA if you need any help.
-}
-- =============================================================================== --
--
module TrainingExercises where
--
import Data.List
import Data.Char
--

{- * 3.1 PATTERN MATCHING -}

type Username = String
type Password = String
type LoggedInTimes = Int

-- Example user tuple used in the following exercies
exampleUser :: (Username, Password, LoggedInTimes)
exampleUser = ("username", "nope", 3)

-- ** TE 3.1.1
--
-- | Write a function that validates user passwords and returns a Bool.
--
-- Rules are simple:
--   1. If the password matches the master password "flasha-ah" (https://www.azlyrics.com/lyrics/queen/flash.html),
--   2. If the first letter is 'H' and the rest of the password is longer than 5 characters
--   3. Otherwise it is a bad password.

te311 :: (Username, Password, LoggedInTimes) -> Bool
te311 (u, "flasha-ah", _) = True
te311 (u, 'H':xs, _) = True
te311 (_, _, _) = False

  

-- ** TE 3.1.2
--
-- | Write a function that transforms a user tuple to a tuple that contains
-- the bool if the password is valid from te311.
--
-- -> Example: te312 ("username", "nope", 3) ==> ("username", False, 3).
-- Bear in mind that you will need the whole input tuple for te311.

te312 :: (Username, Password, LoggedInTimes) -> (Username,Bool,LoggedInTimes)
te312 t@(u, p, n) = (u, te311 t, n) 

-- ** TE 3.1.3
--
-- | Write a function that should return a message based on the validity of the password.
-- Output of te312 is ideal input for this function.
-- -> Example: te313 ("username", True, 3) ==> "Heeey man, glad to see you back."
-- -> Example: te313 ("username", False, 3) ==> "NO! GOD! NO!" (https://giphy.com/gifs/the-office-no-michael-scott-ToMjGpx9F5ktZw8qPUQ)

te313 :: (Username, Bool, LoggedInTimes) -> String
te313 (_, True, _) = "Heeey man, glad to see you back."
te313 (_, False, _) = "NO! GOD! NO!"

-- ** TE 3.1.4 - EXTRA
--
-- | Write a function that goes through a list of users and extracts the LoggedInTimes attribute (list comprehension).
-- After that you should sum the resulting list. (Hint: it has something to do with the `sum` function).

te314 :: [(Username, Password, LoggedInTimes)] -> Int
te314 l = sum [n | (_,_,n) <- l]


{- * 3.2 LOCAL DEFINITION (WHERE & LET) -}

-- ** TE 3.2.1
--
-- | Write a function (using WHERE) that calculates the average value of a list
-- of doubles when its first and last elements are removed.
-- If you have to calculate average of an empty list, throw an error.
-- Ex. te321 [2.0, 4.0, 6.0, 10.0] -> 5.0

te321 :: (Fractional a) => [a] -> a
te321 l
  | len > 0   = sum nl / fromIntegral len
  | otherwise = error "function isn't defined for empty list"
  where len = length nl
        nl  = tail $ init l
        

-- ** TE 3.2.2
--
-- | Write a function (using CASE and WHERE) that takes a string and if that string
-- has one character returns "The string has one character", if two "The string has two characters"
-- or if it has more than two "The string has many characters".
-- If the string is empty, throw an error.
-- Ex. te322 ""    -> error
-- Ex. te322 "A"   -> "The string has one character"
-- Ex. te322 "AA"  -> "The string has two characters"
-- Ex. te322 "AAA" -> "The string has many characters"

te322 :: String -> String
te322 [] = error "String is empty" 
te322 s = "The string has " ++ 
  case s of
  [a]      -> m1
  [a,b]    -> m2
  (a:b:xs) -> m3
  where m1 = "one character"
        m2 = "two characters"
        m3 = "many characters"
        

-- ** TE 3.2.3
--
-- | Define 'quartiles xs' that returns the quartiles (q1,q2,q3) of a given list.
-- The quartiles are elements at the first, second, and third quarter of a list
-- sorted in ascending order. (You can use the built-int 'splitAt' function and
-- the previously defined 'median' function.)
-- quartiles [3,1,2,4,5,6,8,0,7] => (1.5, 4.0, 6.5)

te323 :: (Integral a,Fractional b) => [a] -> (b,b,b)
te323 xs 
  | odd len   = (median l1, median xs, median $ tail l2)
  | otherwise = (median l1, median xs, median l2)
  where len       = length xs
        (l1,l2) = splitAt (len `div` 2) $ sort xs
        

-- you already have this (from the lecture:)
median :: (Integral a, Fractional b) => [a] -> b
median [] = error "median: Empty list"
median xs
  | odd l     = realToFrac $ ys !! h
  | otherwise = realToFrac (ys !! h + ys !! (h-1)) / 2
  where l  = length xs
        h  = l `div` 2
        ys = sort xs


