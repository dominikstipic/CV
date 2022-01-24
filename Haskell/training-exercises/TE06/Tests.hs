import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--
main :: IO ()
main = hspec $ do

    describe "multiplyWith2" $ mapM_
      ( tester "multiplyWith2" multiplyWith2 )
      [
        (0, 0),
        (1, 2),
        (5, 10),
        (-3, -6)
      ]

    describe "add1" $ mapM_
      ( tester "add1" add1 )
      [
        (0, 1),
        (1, 2),
        (5, 6),
        (-3, -2)
      ]

    describe "TE 6.1.3" $ mapM_
      ( tester "te613" te613 )
      [
        (0, 1),
        (1, 3),
        (5, 11),
        (-3, -5),
        (-10, -19)
      ]

    describe "lessThanTen" $ mapM_
      ( tester "lessThanTen" lessThanTen )
      [
        (0, True),
        (1, True),
        (5, True),
        (-3, True),
        (10, False),
        (16, False)
      ]

    describe "apply3Times" $ mapM_
      ( tester3f "apply3Times" apply3Times )
      [
        (multiplyWith2, "multiplyWith2", 0, 0),
        (add1, "add1", 0, 3),
        (multiplyWith2, "multiplyWith2", 5, 40),
        (add1, "add1", 5, 8),
        (multiplyWith2, "multiplyWith2", -3, -24),
        (te613, "te613", 3, 31)
      ]

    describe "filterElem" $ mapM_
      ( tester3f "filterElem" filterElem )
      [
        (lessThanTen, "lessThanTen", [], []),
        (lessThanTen, "lessThanTen", [1, -3, 12, 5, 10, 4], [1, -3, 5, 4]),
        ((>7), "greaterThanSeven", [1, -3, 12, 5, 10, 4], [12, 10])
      ]

    describe "applyOnElem" $ mapM_
      ( tester3f "applyOnElem" applyOnElem )
      [
        (add1, "add1", [], []),
        (add1, "add1", [1, -3, 12, 5, 10, 4], [2, -2, 13, 6, 11, 5]),
        (multiplyWith2, "multiplyWith2", [1, -3, 12, 5, 10, 4], [2, -6, 24, 10, 20, 8])
      ]

    describe "TE 6.2.4" $ mapM_
      ( tester "te624" te624 )
      [
        ([], []),
        ([1, -3, 12, 5, 10, 4], [1, -3, 5, 4]),
        ([152, 24, 31, 12], [])
      ]

    describe "TE 6.2.5" $ mapM_
      ( tester "te625" te625 )
      [
        ([], []),
        ([1, -3, 12, 5, 10, 4], [4, 0, 15, 8, 13, 7]),
        ([152, 24, 31, 12], [155, 27, 34, 15])
      ]

    describe "TE 6.3.1" $ mapM_
      ( tester "te631" te631 )
      [
        ([], []),
        ([1, -3, 12, 5, 10, 4], [1, -3, 5, 4]),
        ([152, 24, 31, 12], [])
      ]

    describe "TE 6.3.2" $ mapM_
      ( tester "te632" te632 )
      [
        ([], []),
        ([1, -3, 12, 5, 10, 4], [4, 0, 15, 8, 13, 7]),
        ([152, 24, 31, 12], [155, 27, 34, 15])
      ]

    describe "TE 6.3.3" $ mapM_
      ( tester "te633" te633 )
      [
        ([], []),
        ([[], [1, -3, 12, 5, 10, 4], [10, 15, 9]], [[], [1, -3, 5, 4], [9]]),
        ([[152, 24, 31, 12], [33]], [[], []])
      ]

    describe "TE 6.4.1" $ mapM_
      ( tester "te641" te641 )
      [
        ([], []),
        ([1, -3, 12, 5, 10, 4], [1, -3, 5, 4]),
        ([152, 24, 31, 12], [])
      ]

    describe "TE 6.4.2" $ mapM_
      ( tester "te642" te642 )
      [
        ([], []),
        ([1, -3, 12, 5, 10, 4], [4, 0, 15, 8, 13, 7]),
        ([152, 24, 31, 12], [155, 27, 34, 15])
      ]

    describe "TE 6.4.3" $ mapM_
      ( tester "te643" te643 )
      [
        ([], []),
        ([[], [1, -3, 12, 5, 10, 4], [10, 15, 9]], [[], [1, -3, 5, 4], [9]]),
        ([[152, 24, 31, 12], [33]], [[], []])
      ]

    where
        tester fname f (x, y) =
          it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
        tester3f fname f (w, x, y, z) =
          it ( fname ++ " " ++ x ++ ", " ++ show y ++ " ==> " ++ show z ) $ f w y `shouldBe` z
