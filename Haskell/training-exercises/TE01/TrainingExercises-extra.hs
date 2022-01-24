-- =============================================================================== --
{- |
  These are the Extra tasks of the first training exercise. You don't have to
  solve them, but are more than welcome to.

  Please, copy the function definitions to your `TrainingExercises.hs` file
  and uncomment tests in `Tests.hs` - do not solve the tasks in this file
  (this is just a template).
-}
-- =============================================================================== --
--
module TrainingExercises where
--
import Data.List
import Data.Char
--
-- ** TE 1.1.4 <- EXTRA
--
-- | Implement a function which takes in two 'Int's, multiplies them and adds 1 to the result.
-- Using our trusty IF-THEN-ELSE check if the result of the multiplication is even and in that
-- case divide it by 2 first. After that add 1 to the resulting number.
--
-- Remember that IF-THEN-ELSE is an expression and always returns a value.
te114 x y = undefined


-- ** TE 1.2.4 <- EXTRA
--
-- | Implement a function which checks if the 'String' is a palindrome. If it is,
-- then return only the first half (if it has odd length then don't include the
-- middle character) otherwise return the 'String' unmodified.
--
-- You can assume that all letters will be lower case.
te124 s = undefined


-- ** TE 1.4.4 <- EXTRA
--
-- | Implement a function which takes two sentences as a tuple and returns a list of
-- common words.
--
-- You can assume that all letters are in lower case. Don't forget that you can use
-- filters in list comprehension.
te134 t = undefined
