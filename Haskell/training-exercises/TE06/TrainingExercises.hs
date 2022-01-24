-- =============================================================================== --
{- |
  Welcome to your sixth Haskell training.

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-06`, create a Merge Request,
  and assign it to your TA.

  Keep in mind that **all exposed functions must have type signatures**
  (helper functions that are defined in local definitions don't)!

  Unlike other exercises, some functions in this exercise don't have names like te6**.
  Instead, they have 'normal' names. You will see why as you solve exercise.
  Do not rename the functions.

  Tthere aren't any extra tasks - all tasks are mandatory.

  As always, ask your TA if you need any help.
-}
-- =============================================================================== --
--
module TrainingExercises where
--
import Data.List
import Data.Char
--


{- * 6.1 Partial application, sections and eta reduction  -}

-- ** TE 6.1.1
-- | Define a function that takes an Int and multiplies it by 2.
-- | (Using eta reduction)

multiplyWith2 :: Int -> Int
multiplyWith2 = (2*)


-- ** TE 6.1.2
-- | Define a function that takes an Int and adds one to it.
-- | (Using eta reduction)

add1 :: Int -> Int
add1 = (1+)


-- ** TE 6.1.3
-- | Define a function that takes an Int and is equivalent to: x*2 + 1.
-- | Achieve that using functions multiplyWith2 and add1.

te613 :: Int -> Int
te613 x = add1 $ multiplyWith2 x


-- | very small lecture :)
-- Aside from their usefulness, partial applications are important because they enable
-- us to write highly readable code. If you wrote te613 correctly you can see that
-- it looks almost like a sentence.

-- ** TE 6.1.4
-- | Define a function that takes an Int and returns true if it is less than 10, false otherwise.
-- | (Using sections)

lessThanTen :: Int -> Bool
lessThanTen = (<10)


{- * 6.2 Higher-order funtions  -}

-- | Solve ALL 6.2.* exercises  *without* using FILTER or MAP.

-- ** TE 6.2.1
-- | Define a function that takes a function and an Int and applies that function
-- | three times to the given number.

apply3Times :: (Int -> Int) -> Int -> Int
apply3Times f x = f $ f $ f x


-- ** TE 6.2.2
-- | Define a recursive function that takes a function and a list of Ints and filters that
-- | list using given function.

filterElem :: (Int -> Bool) -> [Int] -> [Int]
filterElem _ [] = []
filterElem f (x:xs)
  | f x = x : filterElem f xs
  | otherwise = filterElem f xs


-- ** TE 6.2.3
-- | Define a recursive function that takes a function and a list of Ints and applies
-- | given function to each element in list.

applyOnElem :: (Int -> Int) -> [Int] -> [Int]
applyOnElem _ [] = []
applyOnElem f (x:xs) = f x : applyOnElem f xs



-- ** TE 6.2.4
-- Define a function that takes a list of Ints and keeps just those that are less than 10.
-- (Using filterElem, lessThanTen, and eta reduction)

te624 :: [Int] -> [Int]
te624 = filterElem lessThanTen 


-- ** TE 6.2.5
-- (Using applyOnElem, composition, apply3Times, add1 and eta reduction )
-- Define a function that takes a list of Ints and adds 3 to each element in the given list.

te625 :: [Int] -> [Int]
te625 = applyOnElem $ apply3Times add1  



{- * 6.3 Map and filter -}

-- ** TE 6.3.1
-- Rewrite task te624 using built-in filter function

te631 :: [Int] -> [Int]
te631 = filter lessThanTen


-- ** TE 6.3.2
-- Rewrite task te625 using built-in map function

te632 :: [Int] -> [Int]
te632 = map $ apply3Times add1 


-- **  TE 6.3.3
-- Define function that takes a list of lists of Ints and that keeps just those
-- that are less than 10. You must use map, filter, lessThanTen and eta reduction.

te633 :: [[Int]] -> [[Int]]
te633 = map (filter lessThanTen)
 

{- * 6.4 Lambda expressions -}

-- ** TE 6.4.1
-- Rewrite task te631 using lambda expressions
te641 :: [Int] -> [Int]
te641 = filter(\x -> x < 10)


-- ** TE 6.4.2
-- Rewrite task te632 using lambda expressions

te642 :: [Int] -> [Int]
te642 = map(\x -> x + 3)


-- **  TE 6.4.3
-- Rewrite task te633 using lambda expressions
te643 :: [[Int]] -> [[Int]]
te643 = map (filter(\x -> x < 10)) 


