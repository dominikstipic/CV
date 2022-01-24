-- =============================================================================== --
{- |
  Welcome to your fifth Haskell training. Get ready to rumble.
  Where will you let recursion lead you?

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-05`, create a Merge Request,
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

{- * 5.1 Recursive functions with accumulators -}

{-
 - For the following functions, we'd like you to to use the `seq` function to
 - reduce the accumulators to WHNF where needed to avoid building big thunks.
 -}

-- ** TE 5.1.1
--
-- | Define a recursive function with accumulation which finds the
-- | first longest word in a sentence.
-- | Make sure you keep track of the length of the currently longest word
-- | too, so you don't call `length` repeatedly on the same word.

te511 :: String -> String
te511 sen = te511' (words sen) [] 0
 where te511' [] mem _ = mem
       te511' (w:sen) mem len 
         | len < l1  = te511' sen w l1
         | otherwise = te511' sen mem len
         where l1 = length w

-- ** TE 5.1.2
--
-- | Define a recursive function with accumulation which takes a list of
-- | polynomial coefficients and a variable, and calculates the polynomial using
-- | Horner's method (https://en.wikipedia.org/wiki/Horner%27s_method ).
-- | The coefficients are in descending order of variable exponents, so a list
-- | representing the polynomial x^3 - 2x + 3 would be  as  [1, 0, -2, 3].

te512 :: Num a => [a] -> a -> a
te512 [] _ = 0
te512 xs x0 = te512' (tail xs) (head xs)
  where var = x0
        te512' [] s = s
        te512' (a:xs) s = let k = (a + var*s) in k `seq` te512' xs k  
-- ** TE 5.1.3
--
-- | Define a function which computes the population standard deviation of a list of
-- | numbers. To achieve this you need to compute the mean and variance of the list:
-- | do this using recursive functions with accumulation.

te513 :: (Eq a, Floating a) => [a] -> a
te513 [] = error "empty list"
te513 [_] = 0
te513 ys@(x:y:xs) = disp xs 3 m0 d0 
  where m0 = (x + y)/2 
        d0 = ((x-m0)**2 + (y-m0)**2)/2 
        disp [] _ _ d = sqrt d 
        disp (x:xs) n m d =
          let 
            m1 = m + (x-m)/(n) 
            d1 = ( (n-1)*d + (x-m)*(x-m1) )/n
          in disp xs (n+1) m1 d1



-- ** TE 5.1.4
--
-- | An aspiring rollercoaster designer wants to test out his if his newest
-- | creation is safe to ride, and needs your help!
-- | Define a function which takes a list of pairs which describe a section of
-- | the track. The first element will be a String which will be either "even",
-- | "drop", "turn left" or "turn right". The second element will be a number.
-- |
-- | If it's a "drop", the car accelerates as if it was in free-fall (accelerating
-- | 9.81 m/s^2) and the number indicates the height of the drop in meters.
-- | The car maintains its current speed coming into the drop.
-- |
-- | If it's "even", the car decelerates by 0.5 m/s every meter it passes. The
-- | number indicates the length of the even segment. If the car decelerates to
-- | 0 km/s, the track is deemed unsafe as the passengers will become stuck!
-- |
-- | If it's either of the two "turn"s, the number indicates the radius of the
-- | turn in meters. The car will derail if it turns too tightly: it can only
-- | withstand centripetal acceleration of up to and including 5G. And if there
-- | are 3 or more alternating turns directly in a row, the passengers will become
-- | nauseous, which can be unsafe.
-- |
-- | The car starts moving at 20 km/h. The function must return a list indicating
-- | whether the rollercoaster is safe or not. If it is safe, it returns an empty list.
-- | If the rollercoaster is not safe, it returns a list with one element: the
-- | index of the segment of the track where it becomes unsafe.
-- | (Later on, you will learn a much more elegant way of representing a result
-- | which might contain a value, or might contain nothing at all, but this will
-- | do for now.)


te514 :: [(String, Double)] -> [Int]
te514 xs
  | null xs = error "list is empty"
  | otherwise = let vi = 20*10/36 in test xs vi 0 0
  where test [] _ _ _ = []

        test (("drop",h):xs) v0 ind turns = test xs v1 (ind+1) 0
           where v1 = sqrt (v0**2 + 2*9.81*h) 

        test (("even",d):xs) v0 ind turns
           | v1 < 0 = [ind]
           | otherwise = test xs v1 (ind+1) 0
           where v1 = v0 - 0.5*d

        test ((t,r):xs) v ind turns
          | acp > g5 || turns == 2 = [ind]
          | otherwise = test xs v (ind+1) (turns+1)
          where acp = v**2/r
                g5 = 5*9.81 


-- ** TE 5.1.5 - EXTRA
--
-- | Define a recursive function with accumulation which computes the square root
-- | of a given number using Newton's method, with the given number of iterations.
-- | Use the halved original number as an initial guess for the method.
te515 :: (Ord a, Fractional a, Integral b) => a -> b -> a
te515 a iter = newton (a/2) iter 
  where newton x 0 = x
        newton x iter = let x' = 0.5*(x + a/x) in x' `seq` newton x' (iter-1)
           




