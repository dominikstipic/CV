-- =============================================================================== --
{- |
  Welcome to your first Haskell training. Every training exercise ( TE ) has a single
  function in the form of 'texxx', so for training exercise 1.1.1 there will be a
  function 'te111'.

  To start solving the problem simply remove the 'undefined' from the body of a
  function / definition and start writing your solution.

  Before going any further, you should try to update your package index and possibly
  your cabal (cabal is a build tool for Haskell). To update your package index run
  the following command:

  >> cabal new-update

  If that's not working, try

  >> cabal update
  >> cabal install cabal-install

  This will install the latest version of `cabal` which has `new-` commands available.

  To test your solutions you can open the repl / ghci by opening the terminal in the
  folder of your training (e.g. 'C:\Users\minime\Desktop\TE01' if you are on Windows
  or ~/Desktop/TE01 if you are on a Unix system), and typing in the following command
  (without the '>'):

  >> cabal new-repl

  This will download the necessary dependencies (if there are any) and load your
  'TrainingExercises' module into the repl where you can try to call your functions
  with different arguments.

  When you make changes to your code you can type ':r' into the repl to reload your
  code.

  For this training, we have also included the test suite so that you can quickly
  check if your solutions are passing some rudimentary tests.

  To run the test suite, simply type following into your console (you will have to
  exit the repl first, or open another console window):

  >> cabal new-test

  If this fails due to missing hspec, you will have to install it:

  >> cabal new-install hspec

  This will run the tests and print out any errors you may have. Also make sure to
  uncomment tests for extra tasks if you are solving them.

  __YOU CAN ADD YOUR OWN TESTS BUT DO NOT MODIFY OR REMOVE ANY PRE-EXISTING TESTS!__
  __THERE WILL BE SANCTIONS FOR SUCH ACTIONS!__

  PROTIP: You can convert this whole file into a more readable HTML document by
  running the following command:

  >> cabal new-haddock

  And opening the resulting file (you will get the location in the console) with your
  browser.
-}
-- =============================================================================== --

module TrainingExercises where
--
import Data.List
import Data.Char


{- * 1.1 IF-THEN-ELSE & GUARDS -}

-- ** TE 1.1.1
--
-- | Using IF-THEN-ELSE create a function which takes in an 'Int' and divides it by 2
-- if it's even otherwise it adds 1 to it and than divides it by 2.
--
-- You can use 'even' and 'div' functions to make things easier.

te111 :: Int -> Int
te111 n = (if odd n then n + 1 else n) `div` 2
 
-- ** TE 1.1.2
--
-- | Using GUARDS, implement a function which takes in an 'Int' and for numbers from
-- 1 to 3 it returns "one", "two" and "three" respectively, for everything else
-- return "out of range".

te112 :: Int -> String
te112 n
  | n == 1 = "one"
  | n == 2 = "two"
  | n == 3 = "three"
  | otherwise = "out of range" 

-- ** TE 1.1.3
--
-- | Given the following three ranges (not lists) [1,5), [-4,0] and (25,100] by using
-- GUARDS implement a function which takes in an 'Int', checks if the number is
-- within any of the three ranges. If it is, then print out that range, otherwise
-- return "out of range".
--
-- Message should be in the following format: "number is in the [1,5) range"

te113 :: Int -> String
te113 n
  | n >= 1 && n < 5 = "number is in the [1,5) range"
  | n >= -4 && n <= 0 = "number is in the [-4,0] range"
  | n > 25 && n <= 100 = "number is in the (25,100] range"
  | otherwise = "out of range"  

--

{- * 1.2 LIST & STRINGS -}

-- ** TE 1.2.1
--
-- | Implement a function which takes in two lists of the same type and returns the
-- longer one. If the lists are of equal length concatenate them and return that.

te121 :: [a] -> [a] -> [a]
te121 l1 l2
  | length l1 > length l2 = l1
  | length l1 < length l2 = l2
  | otherwise = l1 ++ l2 
  

-- ** TE 1.2.2
--
-- | Implement a function which adds '.' to the end of the 'String' by using ':'
-- (cons operator) and 'reverse' function.

te122 :: String -> String
te122 s = reverse $  '.' : reverse s

-- ** TE 1.2.3
--
-- | Implement a function which removes first three and last three words from the
-- 'String'. If there are not enough words, than return the empty 'String'.
--
-- You can use functions 'words' and 'unwords' to split the 'String' into a list of
-- words.

te123 :: String -> String
te123 s = unwords $ drop 3 $ take (length (words s) - 3) $ words s

--

{- * 1.3 TUPLES & LIST COMPREHENSIONS -}

-- ** TE 1.3.1
--
-- | Using LIST COMPREHENSIONS and TOUPLES implement a function which will take a
-- starting number 'n' and construct infinite list of pairs of numbers and their
-- squares, where numbers are going from 'n' to the infinity in the increments of
-- two.
--
-- Here is an example of such list where 'n' is 3:
-- [(3,9),(5,25),(7,49),(9,81),...]

te131 :: (Enum a, Floating a) => a -> [(a,a)]
te131 n = [(x,x**2) | x <- [n,n+2..]]

-- ** TE 1.3.2
--
-- | Given the lists of 'adjective's and 'noun's, by using LIST COMPREHENSIONS
-- generate a list of all possible book titles in the form of:
--
-- "The {adjective} {noun}"
--
-- and add index to them.
--
-- Example of a single indexed book title in a tuple:
--
-- (2, "The Deadly Bicycle")
--
-- Use the 'adjective' and 'noun' lists in your list comprehension
-- (don't modify them). 'te132' should not take any arguments.

te132 :: [(Integer, String)]
te132 = zip [1..] ["The " ++ x ++ " " ++ y | x <- adjective, y <- noun]


-- | Lists which you should use in your 'te132' implementation.
adjective, noun :: [String]
( adjective, noun ) =
  ( ["Whispering", "Incredible", "Wild", "Deadly"]
  , ["Forest", "Bicycle", "River", "Woman", "Necklace"]
  )


-- ** TE 1.3.3
--
-- | Implement a function which takes in a list of pairs / tuples of 'Int's and
-- returns a list of their sums.
te133 :: [(Int,Int)] -> [Int]
te133 l = [x+y | (x,y) <- l]



-------------EXTRA------------------------------------
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
-- ** TE 1.1.4 <- EXTRA
--
-- | Implement a function which takes in two 'Int's, multiplies them and adds 1 to the result.
-- Using our trusty IF-THEN-ELSE, check if the result of the multiplication is even and in that
-- case divide it by 2 first. After that add 1 to the resulting number.
--
-- Remember that IF-THEN-ELSE is an expression and always returns a value.
te114 :: Int -> Int -> Int
te114 x y = (if even (x*y) then x*y `div` 2 else x*y) + 1


-- ** TE 1.2.4 <- EXTRA
--
-- | Implement a function which checks if the 'String' is a palindrome. If it is,
-- then return only the first half (if it has odd length then don't include the
-- middle character) otherwise return the 'String' unmodified.
--
-- You can assume that all letters will be lower case.

te124 :: String -> String
te124 s = if reverse s == s then take (length s `div` 2) s else s

-- ** TE 1.4.4 <- EXTRA
--
-- | Implement a function which takes two sentences as a tuple and returns a list of
-- common words.
--
-- You can assume that all letters are in lower case. Don't forget that you can use
-- filters in list comprehension.
te134 :: (String, String) -> ([String])
te134 (x,y) = [w1 | w1 <- words x, w2 <- words y, w1==w2]

