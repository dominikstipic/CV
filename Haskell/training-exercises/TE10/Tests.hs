import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
import Data.Char
--

main :: IO ()
main = hspec $ do

    describe "myListEq" $ mapM_
      ( tester2 "myListEq" (==) )
      [
        (
          (Cons 1 Empty),
          (Cons 1 Empty),
          True
        ),
        (
          Empty,
          Empty,
          True
        ),
        (
          (Cons 1 (Cons 2 (Cons 3 Empty))),
          (Cons 3 (Cons 1 (Cons 2 Empty))),
          True
        ),
        (
          (Cons 1 (Cons 2 (Cons 3 Empty))),
          (Cons 1 (Cons 1 (Cons 2 Empty))),
          False
        ),
        (
          (Cons 1 (Cons 2 (Cons 3 Empty))),
          Empty,
          False
        ),
        (
          (Cons 1 (Cons 2 (Cons 3 Empty))),
          (Cons 1 (Cons 2 Empty)),
          False
        )
      ]

    describe "treeToList" $ mapM_
      ( tester "treeToList" treeToList )
      [
        (Ornament, []),
        (Light 1 Ornament Ornament, [1]),
        (Light 1 (Light 2 Ornament Ornament) (Light 4 (Light 3 Ornament Ornament) Ornament), [2, 1, 3, 4]),
        (Light 1 (Light 2 Ornament Ornament) (Light 4 (Light 3 Ornament Ornament) (Light 1 Ornament Ornament)), [2, 1, 3, 4, 1])
      ]

    describe "treeEq" $ mapM_
      ( tester2 "treeEq" (==) )
      [
        ((Light 1 Ornament Ornament), (Light 1 Ornament Ornament), True),
        (
          (Light 1 (Light 2 Ornament Ornament) (Light 4 (Light 3 Ornament Ornament) Ornament)),
          (Light 4 (Light 1 Ornament Ornament) (Light 3 (Light 2 Ornament Ornament) Ornament)),
          True
        ),
        (
          (Light 1 (Light 2 Ornament Ornament) (Light 2 (Light 3 Ornament Ornament) Ornament)),
          (Light 2 (Light 1 Ornament Ornament) (Light 3 (Light 2 Ornament Ornament) Ornament)),
          True
        ),
        (
          (Light 1 (Light 2 Ornament Ornament) (Light 3 (Light 4 Ornament Ornament) Ornament)),
          (Light 5 (Light 6 Ornament Ornament) (Light 7 (Light 8 Ornament Ornament) Ornament)),
          False
        ),
        (
          (Light 1 Ornament (Light 3 (Light 4 Ornament Ornament) Ornament)),
          (Light 5 (Light 6 Ornament Ornament) (Light 7 (Light 8 Ornament Ornament) Ornament)),
          False
        )
      ]

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
        tester2 fname f (x, y, z) =
          it ( fname ++ " " ++ show x ++ " " ++ show y ++ " ==> " ++ show z ) $ f x y `shouldBe` z
