import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--
main :: IO ()
main = hspec $ do

  describe "productList" $ mapM_
    ( tester "productList" product')
    [
      ([4, 0, 3, -9, 1], 12),
      ([1, 2, 3, 4, 5], 120),
      ([0, -3, -2], 1),
      ([], 1)
    ]

  describe "productMaybe" $ mapM_
    ( tester "productMaybe" product')
    [
      (Just 2, 2),
      (Just (-2), 1),
      (Nothing, 1)
    ]

  describe "firstDup" $ mapM_
    ( tester "firstDup" firstDup)
    [
      ([], -1),
      ([2, 3, 1], -1),
      ([3, 3, 2, 2], 1),
      ([5, 2, 3, 5], 3)
    ]

  describe "isPermutationInt" $ mapM_
    ( tester2 "isPermutationInt" isPermutation)
    [
      ([], [], True),
      ([], [1], False),
      ([2, 1, 3], [3, 2, 1], True),
      ([2, 1, 3], [3, 2, 1, 3], False)
    ]

  describe "isPermutationString" $ mapM_
    ( tester2 "isPermutationString" isPermutation)
    [
      ("", "", True),
      ("", "a", False),
      ("bro", "rob", True),
      ("bro", "robb ", False)
    ]

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
        tester2 fname f (x, y, z) =
          it ( fname ++ " " ++ show x ++ ", " ++ show y ++ " ==> " ++ show z ) $ f x y `shouldBe` z
        err fname f x =
          it ( fname ++ " " ++ show x ) $ (f x `seq` pure ()) `shouldThrow` anyException
