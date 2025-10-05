{
  description = "Dev shell with Python 3.12 and uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        uv = pkgs.uv;
        libstdcpp = pkgs.stdenv.cc.cc.lib;
        libz = pkgs.zlib;

        # Python tools
        basedpyright = pkgs.basedpyright;
        black = pkgs.python312Packages.black;
        pyflakes = pkgs.python312Packages.pyflakes;
        isort = pkgs.python312Packages.isort;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "uv-shell";

          packages = [
            python
            uv
            libstdcpp
            libz
            basedpyright
            black
            pyflakes
            isort
          ];

          shellHook = ''
            export UV_PYTHON="${python.interpreter}"
            echo "🔧 uv will use Python at: $UV_PYTHON"
            export LD_LIBRARY_PATH="${libstdcpp}/lib:${libz}/lib:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
}
