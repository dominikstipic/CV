import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
import Data.Char
--
main :: IO ()
main = hspec $ do

    describe "te912a" $ mapM_
      ( tester "te912a" te912 )
      [
        (Pair 1 'a', 1)
      ]

    describe "te912b" $ mapM_
      ( tester "te912b" te912 )
      [
        (Pair "abc" 'a', "abc")
      ]
    
    describe "te912c" $ mapM_
      ( tester "te912c" te912 )
      [
        (Pair [1, 2, 3] "something", [1, 2, 3])
      ]

    describe "te913a" $ mapM_
      ( tester "te913a" te913 )
      [
        (Pair 1 'a', 'a')
      ]

    describe "te913b" $ mapM_
      ( tester "te913b" te913 )
      [
        (Pair "abc" 'a', 'a')
      ]
    
    describe "te913c" $ mapM_
      ( tester "te913c" te913 )
      [
        (Pair [1, 2, 3] "something", "something")
      ]

    describe "te914a" $ mapM_
      ( tester "te914a" te914 )
      [
        (Pair 1 'a', Pair 'a' 1)
      ]

    describe "te914b" $ mapM_
      ( tester "te914b" te914 )
      [
        (Pair "abc" 'a', Pair 'a' "abc")
      ]
    
    describe "te914c" $ mapM_
      ( tester "te914c" te914 )
      [
        (Pair [1, 2, 3] "something", Pair "something" [1, 2, 3])
      ]

    describe "te915" $ mapM_
      ( tester2 "te915" te915 )
      [
        ([Pair 1 'a', Pair 4 'c'], (+ 1), toUpper, [Pair 2 'A', Pair 5 'C']),
        ([], (+ 1), toUpper, []),
        ([Pair 1 'a'], (+ 1), toUpper, [Pair 2 'A']),
        ([Pair 1 'a', Pair 4 'c'], (* 2), toUpper, [Pair 2 'A', Pair 8 'C'])
      ]

    describe "te921a" $ mapM_
      ( tester3 "te921a" te921 )
      [
        (True, Just False, False),
        (True, Just True, True),
        (True, Nothing, True),
        (False, Just False, False),
        (False, Just True, True),
        (False, Nothing, False)
      ]

    describe "te921b" $ mapM_
      ( tester3 "te921b" te921 )
      [
        (0, Just 1000000, 1000000),
        (0, Nothing, 0)
      ]

    describe "te921c" $ mapM_
      ( tester3 "te921c" te921 )
      [
        ("Java", Just "Haskell", "Haskell"),
        ("Java", Nothing, "Java")
      ]

    describe "te922" $ mapM_
      ( tester "te922" te922 )
      [
        ([Just True, Just True, Just True], True),
        ([Just True, Just True, Nothing], False),
        ([Just True, Just True, Just False], False),
        ([Just False, Just True, Just True], False),
        ([Just True, Nothing, Just True], False)
      ]

    describe "te931" $ mapM_
      ( tester "te931" te931 )
      [
        ([Just True, Just True, Just True], [Just 5, Just 5, Just 5]),
        ([Just True, Just True, Nothing], [Just 5, Just 5, Nothing]),
        ([Just True, Just True, Just False], [Just 5, Just 5, Just 1]),
        ([Just False, Just True, Just True], [Just 1, Just 5, Just 5]),
        ([Just True, Nothing, Just True], [Just 5, Nothing, Just 5])
      ]

    describe "te942" $ mapM_
      ( tester "te942" te942 )
      [
        (Nil, 0),
        (Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil), 7),
        (Node 1 (Node 2 (Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil)) Nil) (Node 4 Nil Nil), 14),
        (Node 1 (Node 2 (Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil)) (Node 1 (Node 2 (Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil)) Nil) (Node 4 Nil Nil))) (Node 4 Nil Nil), 28)
      ]

    describe "te943" $ mapM_
      ( tester "te943  " te943 )
      [
        (Nil, Nothing),
        (Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil), Just 4),
        (Node 1 (Node 2 (Node 1 (Node 2 Nil Nil) (Node 5 Nil Nil)) Nil) (Node 4 Nil Nil), Just 5),
        (Node 1 (Node 2 (Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil)) (Node 1 (Node 6 (Node 1 (Node 2 Nil Nil) (Node 4 Nil Nil)) Nil) (Node 4 Nil Nil))) (Node 4 Nil Nil), Just 6)
      ]
      

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
        tester2 fname f (x, f1, f2, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x f1 f2 `shouldBe` y
        tester3 fname f (x, y, z) =
            it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x y `shouldBe` z
