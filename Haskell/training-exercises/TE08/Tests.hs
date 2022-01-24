import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--
main :: IO ()
main = hspec $ do

    describe "te811" $ mapM_
      ( tester "te811" te811 )
      [
        ("dabaccacbacccada", "dabcbacada"),
        ("ababba", "ab"),
        ("ababbaa", "aba"),
        ("ababbba", "ababa"),
        ("abcdefg", "abcdefg")
      ]

    describe "te812" $ mapM_
      ( tester "te812" te812 )
      [
        ("Haskell", 2),
        ("ababba", 3),
        ("ababbaa", 4),
        ("ThIs Is SpartAaA!!", 6)
      ]

    describe "te822" $ mapM_
      ( tester "te822" te822 )
      [
        (Monday, False),
        (Tuesday , False),
        (Wednesday, False),
        (Thursday, True),
        (Friday, False),
        (Saturday, False),
        (Sunday , False)
      ]

    describe "te824" $ mapM_
      ( tester "te824" te824 )
      [
        ([Incoming 15, Outgoing 10, Incoming 3], 8),
        ([Incoming 10, Incoming 10, Incoming 10000], 10020),
        ([Outgoing 10], -10)
      ]

    describe "te825" $ mapM_
      ( tester "te825" te825 )
      [
        ([Incoming 15, Outgoing 10, Incoming 3], True),
        ([Incoming 10, Incoming 10, Incoming 10000], True),
        ([Outgoing 10], False),
        ([Outgoing 10, Incoming 15, Incoming 3], False)
      ]

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
