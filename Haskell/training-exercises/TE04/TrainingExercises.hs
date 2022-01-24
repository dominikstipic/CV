-- =============================================================================== --
{- |
  Welcome to your fourth Haskell training. Get ready to rumble.
  Where will you let recursion lead you?

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-04`, create a Merge Request,
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

{- * 4.1 Recursive functions -}

-- ** TE 4.1.1
--
-- | Define a recursive function that calculates the minimal number of moves needed to complete a game
-- | of Towers of Hanoi with n disks. 
-- | 
-- | In case you don't know about that game, take a look here: 
-- | https://en.wikipedia.org/wiki/Tower_of_Hanoi#Recursive_solution
--
te411 :: Int -> Int
te411 0 = 0
te411 n
  | n >= 0 = 1 + 2 * te411 (n-1) 
  | otherwise = error "game is not defined for negative numbers"

             


-- ** TE 4.1.2
--
-- | Define a recursive function that calculates the greatest common divisor of two given numbers.
--

te412 :: Int -> Int -> Int
te412 a 0 = a
te412 a b = te412 b (a `mod` b) 


-- ** TE 4.1.3
--
-- | Define a recursive function that returns the last element of a list.
-- | What do you think should happen with an empty list?
--

te413 :: [a] -> a
te413 [] = error "Not defined for empty list"
te413 [x] = x
te413 (x:xs) = te413 xs 

-- ** TE 4.1.4
--
-- | You have seen a Quick Sort implementation on the lecture. Now is the time to implement Merge Sort.
-- | You are not allowed to use list comprehension here!
--

a = [6,5,3,1,8,7,2,4]
te414 :: Ord a => [a] -> [a]
te414 [] = []
te414 [x] = [x]
te414 [x,y]
  | x > y = [y,x]
  | otherwise = [x,y] 
te414 xs =  merge (te414 x) (te414 y) 
  where len = length xs
        (x,y) = splitAt (len `div` 2) xs 
        merge [] ys = ys
        merge xs [] = xs
        merge l1@(x:xs) l2@(y:ys)
          | y < x = y : (merge l1 ys)
          | otherwise = x : (merge xs l2)    




-- ** TE 4.1.5 - EXTRA
--
-- | Now you have written 2 different efficient sorting algorithms. Let's write something worse!
-- | Write an Insertion sort function.
-- | List comprehensions are not alowed once again! 
--

insert' :: (Eq a,Ord a)  => [a] -> a -> [a]
insert' lis x = insert1 lis x [] 
  where insert1 [] acc tmp = (reverse tmp) ++ [acc]
        insert1 (y:ys) acc tmp
          | acc > y   =  reverse tmp ++ [acc,y] ++ ys
          | otherwise = insert1 ys acc (y:tmp)

te415 :: Ord a => [a] -> [a]
te415 [] = []
te415 (x:xs) = sort1 xs [x] 
  where sort1 [] l = reverse l 
        sort1 (x:xs) l@(y:ys)
          | x < y     = sort1 xs (insert' l x)
          | otherwise = sort1 xs (x:l)  
  

  {- * 4.2 Corecursion -}

-- ** TE 4.2.1
--
-- | Write your own definition of the cycle function.
--


te421 :: [a] -> [a]
te421 xs = xs ++ te421 xs


