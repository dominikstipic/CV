import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--
main :: IO ()
main = hspec $ do

    describe "TE 3.1.1" $ mapM_
      ( tester "te311" te311 )
      [
        (("username", "flasha-ah", 3), True),
        (("freddie", "Haskell", 5), True),
        (("marko", "javaiscool", 10), False)
      ]

    describe "TE 3.1.2" $ mapM_
      ( tester "te312" te312 )
      [
        (("username", "flasha-ah", 3), ("username", True, 3)),
        (("freddie", "Haskell", 5), ("freddie", True, 5)),
        (("marko", "javaiscool", 10), ("marko", False, 10))
      ]

    describe "TE 3.1.3" $ mapM_
      ( tester "te313" te313 )
      [
        (("username", True, 3), "Heeey man, glad to see you back."),
        (("marko", False, 10), "NO! GOD! NO!")
      ]

    describe "TE 3.1.4" $ mapM_
      ( tester "te314" te314 )
      [
        ([("username", "flasha-ah", 3), ("username", "flasha-ah", 2), ("username", "flasha-ah", 1)], 6),
        ([], 0)
      ]

    describe "TE 3.2.1" $ mapM_
      ( tester "te321" te321 )
      [
        ([2.0,4.0,6.0,10.0], 5.0),
        ([1.0,2.0,3.0], 2.0)
      ]

    describe "TE 3.2.1" $ mapM_
      ( err "te321" te321 )
      [
        [],
        [1],
        [1,2]
      ]

    describe "TE 3.2.2" $ mapM_
      ( tester "te322" te322 )
      [
        ("h","The string has one character"),
        ("jj","The string has two characters"),
        ("rtz","The string has many characters"),
        ("asdfoio","The string has many characters")
      ]

    describe "TE 3.2.2" $ mapM_
      ( err "te322" te322 )
      [
        ""
      ]

    describe "TE 3.2.3" $ mapM_
      ( tester "te323" te323 )
      [
        ([3,1,2,4,5,6,8,0,7], (1.5, 4.0, 6.5)),
        ([3,2,5,6,4,0,7], (2.0, 4.0, 6.0)),
        ([1,2,3], (1.0, 2.0, 3.0))
      ]

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
        err fname f x =
          it ( fname ++ " " ++ show x ) $ (f x `seq` (pure ())) `shouldThrow` anyException
