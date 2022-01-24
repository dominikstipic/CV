{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -fno-warn-implicit-prelude #-}
module Paths_TE11 (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/home/doms/.cabal/bin"
libdir     = "/home/doms/.cabal/lib/x86_64-linux-ghc-8.0.2/TE11-0.1.0.0-4O2RlWTPPVI1uzwH6wzNYo"
dynlibdir  = "/home/doms/.cabal/lib/x86_64-linux-ghc-8.0.2"
datadir    = "/home/doms/.cabal/share/x86_64-linux-ghc-8.0.2/TE11-0.1.0.0"
libexecdir = "/home/doms/.cabal/libexec"
sysconfdir = "/home/doms/.cabal/etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "TE11_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "TE11_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "TE11_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "TE11_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "TE11_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "TE11_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
