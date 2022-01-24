import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--
main :: IO ()
main = hspec $ do

    describe "TE 4.1.1" $ mapM_
      ( tester1 "te411" te411 )
      [
        (1, 1),
        (3, 7),
        (4, 15),
        (5, 31),
        (6, 63)
      ]

    describe "TE 4.1.2" $ mapM_
      ( tester2 "te412" te412 )
      [
        (1, 0, 1),
        (42, 38, 2),
        (137, 256, 1),
        (128, 1024, 128)
      ]
        
    describe "TE 4.1.3a" $ mapM_
      ( tester1 "te413" te413 )
      [
        ("Haskell", 'l'),
        ("FER", 'R')
      ]

    describe "TE 4.1.3b" $ mapM_
      ( tester1 "te413" te413 )
      [
        (["This", "is", "a", "random", "sentence"], "sentence")
      ]

    describe "TE 4.1.3c" $ mapM_
      ( tester1 "te413" te413 )
      [
        ([1, 2, 3, 4, 5], 5)
      ]

    describe "TE 4.1.3err" $ mapM_
      ( err "te413" te413 )
      [
        ""
      ]

    describe "TE 4.1.4" $ mapM_
      ( tester1 "te414" te414 )
      [
        ([], []),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 6, 2, 5, 1, 2], [1, 1, 2, 2, 5, 6]),
        ([2, 7, 3, 6, 4, 1, 0, 8, 9, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      ]

    describe "TE 4.1.5" $ mapM_
      ( tester1 "te415" te415 )
      [
        ([], []),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 6, 2, 5, 1, 2], [1, 1, 2, 2, 5, 6]),
        ([2, 7, 3, 6, 4, 1, 0, 8, 9, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      ]

    describe "TE 4.2.1" $ mapM_
      ( testerX "te421" te421 )
      [
        ("a", "aaaaaaaaaa"),
        ("ab", "ababababab"),
        ("abc", "abcabcabca")
      ]    

    where
      tester1 fname f (x, y) =
        it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
      tester2 fname f (x, y, z) =
        it ( fname ++ " " ++ show x ++ ", " ++ show y ++ " ==> " ++ show y ) $ f x y `shouldBe` z
      testerX fname f (x, y) =
        it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ take 10 (f x) `shouldBe` y
      err fname f x =
        it ( fname ++ " " ++ show x ) $ (f x `seq` pure ()) `shouldThrow` anyException
