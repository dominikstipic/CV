-- =============================================================================== --
{- |
  Welcome to your seventh Haskell training.

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-07`, create a Merge Request,
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
import Data.Function
--

{- * 7.1 Function composition  -}

-- ** TE 7.1.1
--
-- | You are given a list of tuples containing a username and the number of times
-- the user has logged in.
-- Write a function that sums the value of the logged in attribute for users whose
-- username is shorter than 7 characters.
-- You are not allowed to use list comprehensions and the function must be eta reduced.
--
-- -> Example: [("Marvin", 4), ("Kobra1997", 7), ("Ryder", 1)] ==> 5.
te711 :: [(String, Int)] -> Int
te711 = sum . map snd . filter ((7>) . length . fst)

-- ** TE 7.1.2
--
-- | Define a function which takes a list of floats and sorts the squares
-- of all the elements.
-- You are not allowed to use list comprehensions and the function must be eta reduced.
--
-- -> Example: [2.0, -3.0, 4.0, 1.0] ==> [1.0, 4.0, 9.0, 16.0].

te712 :: (Floating a, Ord a) => [a] -> [a]
te712 = sort . map (**2)

-- ** TE 7.1.3
--
-- | Define a function that takes a list and returns only the elements in non-even positions.
-- You are not allowed to use list comprehensions and the function must be eta reduced.
--
-- -> Example: [7, 0, 2, 3, 1, 6, 7] ==> [7, 2, 1, 7].

te713 :: [a] -> [a]
te713 = map snd . filter (even . fst) . zip [0..]


{- * 7.2 Useful higher order functions -}

-- ** TE 7.2.1
--
-- | You are given a list of tuples containing a username and the number of times
-- the user has loggen in.
-- Define a function which returns the list of users sorted by the number of times
-- they logged in.
-- The function must be eta reduced and you are allowed to use functions from Data.List and Data.Ord.
--
-- -> Example: [("qui-gon", 3), ("deathsticks", 4), ("Ani", 2)] -> [("deathsticks", 4), ("qui-gon", 3), ("Ani", 2)]

te721 :: [(String, Int)] -> [(String, Int)]
te721 = sortBy (comparing (negate . snd)) 


-- ** TE 7.2.2
--
-- | You are given a list of tuples containing a username and the number of times
-- the user has loggen in.
-- Define a function which groups separates the users who logged in more than 20 times from those
-- who haven't.
-- The function must be eta reduced and you are allowed to use functions from Data.List and Data.Ord.
--
-- -> Example: [("tommy", 23), ("mark", 5), ("lisa", 17), ("denny", 40)] -> ([("tommy",23),("denny",40)],[("mark",5),("lisa",17)])

te722 :: [(String, Int)] -> ([(String, Int)], [(String, Int)])
te722 = partition ((>20) . snd)

-- ** TE 7.2.3
--
-- Define a function which recieves two lists and returns the result of their elementwise multiplication.
-- The function must be eta reduced and you are allowed to use functions from Data.List and Data.Ord.
--
-- -> Example: [4.0, 5.0, 6.0, 1.0] -> [1.0, 3.0, 6.0, 9.0] -> [4.0, 15.0, 36.0, 9.0]

te723 :: Num a => [a] -> [a] -> [a]
te723 = zipWith (*)
