import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--

infix 1 `shouldBeAround`
shouldBeAround :: (HasCallStack, Show a, Floating a, Ord a) => a -> a -> Expectation
x `shouldBeAround` y = x `shouldSatisfy` (\v -> abs (x - y) < 1e-8)

main :: IO ()
main = hspec $ do

    describe "TE 5.1.1" $ mapM_
      ( tester1 "te511" te511 )
      [
        ("Lorem ipsum", "Lorem"),
        ("This is not a particularly long sentence", "particularly"),
        ("It's good to read these tests. They might clue you into the faults in your program!", "program!"),
        ("same size word what will this do", "same")
      ]

    describe "TE 5.1.2" $ mapM_
      ( tester2 "te512" te512 )
      [ 
        ([], 999999, 0),
        ([1], 999999999, 1),
        ([1, 1], 9999999, 10000000),
        ([1, -1, 10], 1024, 1047562)
      ]
        
    describe "TE 5.1.3a" $ mapM_
      ( testerf1 "te513" te513 )
      [
        ([1], 0),
        ([10], 0),
        ([1,2], 0.5),
        ([2,1], 0.5),
        ([1,2,3], 0.816496580927726),
        ([1,2,3,4], 1.118033988749895)
      ]

    describe "TE 5.1.3err" $ mapM_
      ( err "te513" te513 )
      [
        []
      ]

    describe "TE 5.1.4a" $ mapM_
      ( tester1 "te514" te514 )
      [
        ([("drop", 10)], []),
        ([("drop", 10), ("turn left", 4.7)], []),
        ([("drop", 10), ("turn left", 4.6)], [1]),
        ([("drop", 10), ("turn left", 4.6), ("turn right", 1)], [1]),
        ([("turn left", 10), ("turn right", 10), ("turn left", 10)], [2]),
        ([("turn right", 10), ("turn left", 10), ("turn right", 10)], [2]),
        ([("even", 11.1111)], []),
        ([("even", 11.1112)], [0]),
        ([("drop", 5), ("drop", 5), ("turn left", 4.7)], []),
        ([("drop", 5), ("drop", 5), ("turn left", 4.6)], [2]),
        ([("drop", 5), ("drop", 5), ("turn right", 4.7)], []),
        ([("drop", 5), ("drop", 5), ("turn right", 4.6)], [2]),
        ([("drop", 10), ("even", 15 * 2)], []),
        ([("drop", 5), ("drop", 5), ("even", 15 * 2)], []),
        ([("drop", 10), ("even", 15.07 * 2)], [1]),
        ([("drop", 5), ("drop", 5), ("even", 15.07 * 2)], [2])
      ]

    describe "TE 5.1.4err" $ mapM_
      ( err "te514" te514 )
      [
        []
      ]

    describe "TE 5.1.5" $ mapM_
      ( testerf2 "te515" te515 )
      [
        (4, 10, 2),
        (100, 10, 10),
        (121, 10, 11),
        (10000, 10, 100),
        (100, 0, 50),
        (100, 1, 26.0)
      ]

    where
      tester1 fname f (x, y) =
        it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
      testerf1 fname f (x, y) =
        it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBeAround` y
      tester2 fname f (x, y, z) =
        it ( fname ++ " " ++ show x ++ ", " ++ show y ++ " ==> " ++ show y ) $ f x y `shouldBe` z
      testerf2 fname f (x, y, z) =
        it ( fname ++ " " ++ show x ++ ", " ++ show y ++ " ==> " ++ show y ) $ f x y `shouldBeAround` z
      testerX fname f (x, y) =
        it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ take 10 (f x) `shouldBe` y
      err fname f x =
        it ( fname ++ " " ++ show x ) $ (f x `seq` pure ()) `shouldThrow` anyException
