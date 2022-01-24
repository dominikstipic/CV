import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--
main :: IO ()
main = hspec $ do

    describe "te711" $ mapM_
      ( tester "te711" te711 )
      [
        ([("Marvin", 4), ("Kobra1997", 7), ("Ryder", 1)], 5),
        ([("Darth Jar-Jar", 9)], 0),
        ([], 0),
        ([("a", 1), ("b", 100), ("cdefghijk", 100), ("lmnoprsqtuv", 1)], 101)
      ]

    describe "te712" $ mapM_
      ( tester "te712" te712 )
      [
        ([2.0, -3.0, 4.0, 1.0], [1.0, 4.0, 9.0, 16.0]),
        ([], []),
        ([10.0], [100])
      ]

    describe "te713" $ mapM_
      ( tester "te713" te713 )
      [
        ([7, 0, 3, 2, 1, 6, 7], [7, 3, 1, 7]),
        ([5], [5]),
        ([1, 2], [1]),
        ([], [])
      ]

    describe "te721" $ mapM_
      ( tester "te721" te721 )
      [
        ([("qui-gon", 3), ("deathsticks", 4), ("Ani", 2)], [("deathsticks", 4), ("qui-gon", 3), ("Ani", 2)]),
        ([], []),
        ([("qui-gon", 3)], [("qui-gon", 3)])
      ]

    describe "te722" $ mapM_
      ( tester "te722" te722 )
      [
        ([("tommy", 23), ("mark", 5), ("lisa", 17), ("denny", 40)], ([("tommy",23),("denny",40)],[("mark",5),("lisa",17)])),
        ([("tommy", 23), ("denny", 40)], ([("tommy",23),("denny",40)],[])),
        ([("mark", 5), ("lisa", 17)], ([],[("mark",5),("lisa",17)])),
        ([], ([],[]))
      ]

    describe "te723" $ mapM_
      ( tester2 "te723" te723 )
      [
        ([4.0, 5.0, 6.0, 1.0], [1.0, 3.0, 6.0, 9.0], [4.0, 15.0, 36.0, 9.0]),
        ([], [], []),
        ([1.5, 2.0], [5.0, 4.0], [7.5, 8.0])
      ]

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
        tester2 fname f (x, y, z) =
          it ( fname ++ " " ++ show x ++ ", " ++ show y ++ " ==> " ++ show z ) $ f x y `shouldBe` z
        testerp fname f (x, y, z) =
          it ( fname ++ " " ++ show y ++ " ==> " ++ show z ) $ f x y `shouldBe` z
        err fname f x =
          it ( fname ++ " " ++ show x ) $ (f x `seq` pure ()) `shouldThrow` anyException
