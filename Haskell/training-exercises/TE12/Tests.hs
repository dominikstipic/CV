import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
import Data.List

--
main :: IO ()
main = hspec $ do

  describe "te1211" $ mapM_
    ( tester "te1211" te1211)
    [
      ("9567", 1900),
      ("2345", 2500),
      ("4284", 1800),
      ("1234", 0)
    ]

  describe "te1211m" $ mapM_
    ( tester "te1211m" te1211m)
    [
      ("9567", 1900),
      ("2345", 2500),
      ("4284", 1800),
      ("1234", 0)
    ]

  describe "te1212" $ mapM_
    ( sortedTester "te1212" getIds)
    [
      (Tangerine, ["2345","4284"]),
      (Kiwi, ["9567"]),
      (Strawberry, ["4284","9567"])
    ]

  describe "multiplyBy2" $ mapM_
    ( tester2NoShow "multiplyBy2" runSM)
    [
      (multiplyBy2 >> get, 2, 4),
      (multiplyBy2 >> get, 10, 20),
      (multiplyBy2 >> get, -2, -4),
      (multiplyBy2 >> get, 0, 0)
    ]

  describe "add3" $ mapM_
    ( tester2NoShow "add3" runSM)
    [
      (add3 >> get, 2, 5),
      (add3 >> get, 10, 13),
      (add3 >> get, -2, 1),
      (add3 >> get, 0, 3),
      (add3 >> get, -3, 0)
    ]

  describe "te1223" $ mapM_
    ( tester "te1223" te1223)
    [
      (2, 11),
      (10, 43),
      (-2, -5),
      (0, 3),
      (-3, -9)
    ]

  describe "te1224" $ mapM_
    ( tester "te1224" te1224)
    [
      (2, 11),
      (10, 43),
      (-2, -5),
      (0, 3),
      (-3, -9)
    ]

  describe "te1225" $ mapM_
    ( tester "te1225" te1225)
    [
      ([], []),
      ([1, 2], [1, 1, 1, 2]),
      ([-1, 0, 4, 6, 9], [-1, -1, -1, 0, 4, 6, 9, 9, 9]),
      ([1, 1, 2, 3, 5], [1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 5, 5, 5])
    ]

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
        sortedTester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ (sort $ f x) `shouldBe` y
        tester2 fname f (x, y, z) =
          it ( fname ++ " " ++ show x ++ ", " ++ show y ++ " ==> " ++ show z ) $ f x y `shouldBe` z
        tester2NoShow fname f (x, y, z) =
          it ( fname ++ " " ++ ", " ++ show y ++ " ==> " ++ show z ) $ f x y `shouldBe` z
        err fname f x =
          it ( fname ++ " " ++ show x ) $ (f x `seq` pure ()) `shouldThrow` anyException
