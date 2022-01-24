-- =============================================================================== --
{- |
  Welcome to your twelfth Haskell training.

  You should know the drill by now - solve the exercises in `TrainingExercises.hs`,
  run the tests with Cabal, push to `training-10`, create a Merge Request,
  and assign it to your TA.

  Keep in mind that **all exposed functions and instance definitions must have
  type signatures**.
  (helper functions that are defined in local definitions don't)!

  Also there aren't any extra tasks - all tasks are mandatory.

  Don't worry about the performance of your code.

  As always, ask your TA if you need any help.

-}
-- =============================================================================== --
--
module TrainingExercises where
--
import Data.Char
import Data.List
import Data.Ord
import Control.Monad
import Data.Maybe
--

data Farmer = Farmer {
  forename :: String,
  surname  :: String,
  farmerId :: Maybe String } deriving (Show,Eq)

data Fruit =  Tangerine | Kiwi | Strawberry deriving (Read,Show,Eq)

data FruitStock = FruitStock {
    fruit :: Fruit, -- type of fruit
    amount :: Int, -- in kg
    farmer :: Maybe Farmer -- farmer that produced this stock of fruit
    } deriving (Show,Eq)

john = Farmer "John" "Doe" (Just "2345")
jane = Farmer "Jane" "Fox" (Just "9567")
ann  = Farmer "Ann"  "Doe" (Just "4284") 
mark = Farmer "Mark" "Adley" Nothing 

stock1 = FruitStock Tangerine  2500 (Just john) 
stock2 = FruitStock Tangerine  4000 (Nothing) 
stock3 = FruitStock Tangerine  1000 (Just ann) 
stock4 = FruitStock Kiwi       550  (Just mark)
stock5 = FruitStock Kiwi       1000 (Just jane)
stock6 = FruitStock Strawberry 800  (Just ann) 
stock7 = FruitStock Strawberry 700  (Just mark)
stock8 = FruitStock Strawberry 900  (Just jane)
stock9 = FruitStock Strawberry 400  (Nothing)

company_stock = [stock1,stock2,stock3,stock4,stock5,stock6,stock7,stock8,stock9]

-- Above we have a database of one company's stocks of fruit.
-- The company did a poor job maintaining the database so some information is missing.
-- Our job is to find the requested information.

 

{- * 12.1 Monads 1  -}

-- ** TE 12.1.1

-- Write two functions that extract information of how many kilograms of fruit the farmer with a certain ID produced.
-- Implement the function te1211 without using monads and te1211m using monads.
-- Feel free to write helper functions if necessary.
-- ex te1211 "9567" -> 1900

te1211 :: String -> Int
te1211 key = sum . map amount $ filter predicate company_stock
  where predicate s = case farmer s of
                        Just f -> case farmerId f of
                          Just k -> k == key 
                          Nothing -> False
                        Nothing -> False


te1211m :: String -> Int
te1211m key = sum . map amount $ filter (maybe False id . predicate) company_stock
  where predicate s = farmer s >>= farmerId >>= return . (key ==)
  


-- **** I hope you could see from this example how much cleaner the code is with monads. ****


-- ** TE 12.1.2

-- First, write a function which gets a fruit as input and returns a list of ID numbers of all the farmers which produced that fruit.
-- Then, write a function which gets the name of the fruit from console and prints out that list.
-- Inputs will always be one of defined fruits. (Tangerine, Kiwi, Strawberry) 
-- Use monads whenever possible. Skip farmers that dont have an ID number.
-- ex te1212 "Strawberry" -> ["4284","9567"] or ["9567","4284"] (order is unimportant)

getIds :: Fruit -> [String]
getIds f = filter (""/=) . map (maybe "" id . toId) $ filter ((f==) . fruit) company_stock
  where toId s = farmer s >>= farmerId

te1212 :: IO ()
te1212 = do
  input <- getLine
  let fruit = read input :: Fruit
  mapM_ print $ getIds fruit 


{- * 12.1 Monads 2  -}

data SM s a = SM (s -> (s, a))

instance Functor (SM s) where
  fmap f (SM a) = SM $ \s -> let (s', a') = a s in (s', f a') 

instance Applicative (SM s) where
  pure a = SM (\s -> (s,a))
  SM f <*> (SM a) = SM $ \s -> 
    let (s',f') = f s
        (s'', a') = a s'
    in (s'', f' a')

instance Monad (SM s) where
  return a = SM (\s -> (s, a)) 
  
  SM sm0 >>= fsm1 = SM $ \s0 ->
    let (s1,a1) = sm0 s0  -- left computation on the state
        SM sm1 = fsm1 a1  -- the computation of the "right monad"
        (s2,a2) = sm1 s1  -- right computation on the state
    in (s2,a2)
  
-- To remind you what it looks like we defined the state monad again.

get :: SM s s
get = SM (\s -> (s, s))

set :: s -> SM s ()
set a = SM (\_ -> (a, ()))

-- Reminder of what get and set functions look like.



-- ** TE 12.2.1

-- Write function that "runs the monad" and returns just the value of the computation.

runSM :: SM s a -> s -> a
runSM (SM fun) = snd . fun


-- ** TE 12.2.2

-- Write a state monad which multiplies the state (state is of type Int in this case) by two.
multiplyBy2 :: SM Int ()
multiplyBy2 = SM (\s -> (2*s, ()) )


-- ** TE 12.2.3

-- Write a state monad which adds 3 to the state (state is of type Int in this case).
add3 :: SM Int ()
add3 = SM (\s -> (3+s, ())) 

-- ** TE 12.2.3

-- Write a function (using functions add3 and multiplyBy2) which takes an Int
-- and does an operation equivalent to: x*4 + 3
-- ex. te1223 2 = 11  

te1223 :: Int -> Int
te1223 = runSM (multiplyBy2 >> multiplyBy2 >> add3 >> get)


-- ** TE 12.2.4

-- Write the same function as in TE 12.2.3 but this time using the "do" notation.
-- You will probably need a helper function here.

te1224 :: Int -> Int
te1224 = runSM foo

foo :: SM Int Int
foo = do
  multiplyBy2
  multiplyBy2
  add3
  get
  

-- if you wrote it correctly, the helper function should look something like this:

--foo = do
--  multiplyBy2
--  multiplyBy2
--  add3
--  get

-- As you can see here we are just listing all computations that need to be done.


-- ** TE 12.2.5

-- Write a function (using list monads) that takes a list of Int and returns a list
-- with each odd element repeated three times.
-- ex te1225 [1,2] -> [1,1,1,2]

te1225 :: [Int] -> [Int]
te1225 xs = do
  x <- xs
  c <- fun x 
  return c

fun :: Int -> [Int]
fun x 
  | odd x     = [x,x,x]
  | otherwise = [x]  







