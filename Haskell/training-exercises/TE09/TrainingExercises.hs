-- =============================================================================== --
{- |
  Welcome to your ninth Haskell training.

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-09`, create a Merge Request,
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

{- * 9.1 Parametrized Types  -}

-- ** TE 9.1.1
--
-- | Define a new parameterised data type `Pair` that has 2 parameters, a and b.
-- The data type should contain a single type constructor named Pair.
--
-- Make sure to derive `Show` and `Eq`.
--
data Pair a b = Pair a b deriving (Show,Eq)

-- ** TE 9.1.2
--
-- | Define a function which takes a Pair and returns its first element.
--
-- -> Example: Pair 1 'a' ==> 1

te912 :: Pair a b -> a
te912 (Pair x _ ) = x

-- ** TE 9.1.3
--
-- | Define a function which takes a Pair and returns its second element.
--
-- -> Example: Pair 1 'a' ==> 'a'

te913 :: Pair a b -> b
te913 (Pair _ y) = y

-- ** TE 9.1.4
--
-- | Define a function which takes a Pair and swaps the order of its elements.
--
-- -> Example: Pair 1 'a' ==> Pair 'a' 1

te914 :: Pair a b -> Pair b a
te914 (Pair x y) = Pair y x

-- ** TE 9.1.5
--
-- | Define a function which takes a list of Pairs and two functions. The function should
-- apply the first function to the first element of all the pairs, and apply the second one
-- to all the second elements.
--
-- -> Example: [Pair 1 'a', Pair 4 'c'] inc toupper ==> [Pair 2 'A', Pair 5 'C']

te915 :: [Pair a b] -> (a -> a) -> (b -> b) -> [Pair a b]
te915 pairs fun1 fun2 = map (\(Pair x y) -> Pair (fun1 x) $ fun2 y) pairs  

{- * 9.2 Maybe Type  -}

-- ** TE 9.2.1
--
-- | Define a function that takes a Maybe data type and returns its value if it exists and a default
-- value otherwise.
--
-- -> Example: True (Just False) ==> False
-- -> Example: True Nothing      ==> True
-- -> Example: 0 Nothing         ==> 0

te921 :: a -> Maybe a -> a
te921 x Nothing  = x
te921 _ (Just y) = y 

-- ** TE 9.2.2
--
-- | Define a function that determines if a student has solved all of his Training Exercises.
-- The function will get a list of results as input and should return a boolean.
-- Every element in the list will represent the result of a single TE, and can be either True
-- or False, but it is also possible that the student didn't submit the TE at all.
--
-- -> Example: [Just True, Just True, Just True] ==> True
-- -> Example: [Just True, Just True, Nothing]   ==> False

te922 :: [Maybe Bool] -> Bool
te922 = and . map (te921 False)  


{- * 9.3 Fmap  -}

-- ** TE 9.3.1
--
-- | Define a function that calculates the grade for solved Training Exercises. The function
-- argument is a list of Training Exercises (Maybe Bools) where True means that the exercise
-- is solved correctly. In case that an exercise is not solved (Nothing), return Nothing.
--
-- P.S.
-- As we are really, very, extremely strict while grading Training Exercises,
-- you can assume that every training exercise that got a passing grade will be
-- graded as 5, while all the failing ones will be graded as 1.
--
-- -> Example: [Just True, Just True, Just True] ==> [Just 5, Just 5, Just 5]
-- -> Example: [Just True, Just False, Nothing]   ==> [Just 5, Just 1, Nothing]

te931 :: [Maybe Bool] -> [Maybe Int]
te931 = map (fmap giveGrade) 
  where giveGrade True = 5
        giveGrade _    = 1

{- * 9.4 Recursive Data Structures  -}

-- ** TE 9.4.1
--
-- | Take a look at the following definition for a Tree:

data Tree a = Nil
            | Node { value :: a,
                    left  :: Tree a,
                    right :: Tree a }
  deriving Show

-- ** TE 9.4.2
--
-- | Write a function that calculates the sum of all nodes in a given Tree.
--
-- -> Example: Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil) ==> 7

te942 :: Num a => Tree a -> a
te942 Nil  = 0
te942 tree = value tree + te942 (left tree) + te942 (right tree)

-- ** TE 9.4.3
--
-- | Write a function that returns the maximum node in a given Tree.
--
-- -> Example: Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil) ==> Just 4

-- (>) :: Ord a => a -> a -> Bool
te943 :: Ord a => Tree a -> Maybe a
te943 Nil          = Nothing
te943 (Node x l r) =  max (Just x) $ max leftMax rightMax
  where leftMax  = te943 l
        rightMax = te943 r


