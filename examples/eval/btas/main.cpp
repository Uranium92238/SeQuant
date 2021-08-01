//
// Created by Bimal Gaudel on 7/31/21.
//
#include <clocale>
#include <iostream>

#include <btas/btas.h>
#include <SeQuant/core/op.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>

#include "examples/eval/btas/scf_btas.hpp"
#include "examples/eval/calc_info.hpp"

///
/// excitation (2 = ccsd (default) through 6 supported)
/// fock(/eri) tensor data file name (default "fock.dat"/"eri.dat")
/// eri(/fock) tensor data file name (default "fock.dat"/"eri.dat")
/// \note Format of data files:
/// ----------------------------
/// size_t size_t size_t         # rank, nocc, nvirt
/// double                       # data ------
/// ...                          # data       |
/// ...                          # ....       |  no. of double entries =
/// (nocc+nvirt)^rank
/// ...                          # data       |
/// double                       # data ------
/// ----------------------------
/// \note The rank of fock tensor is 2
/// \note The rank of eri tensor is 4
/// \note The nocc and nvirt in both files must match
///
/// \todo Use markdown file formats such as yml for input of calculation params.
///
int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "\nHelp:\n"
              << "<executable> <config_file> <fock_or_eri_file> "
                 "<eri_or_fock_file> [<output_file>]"
              << "\n\n"
              << "Config file format\n"
              << "----\n"
              << sequant::eval::ParseConfigFile{}.help() << "----\n";
    return 1;
  }

  std::setlocale(LC_ALL, "en_US.UTF-8");
  std::wcout.imbue(std::locale("en_US.UTF-8"));
  std::wcerr.imbue(std::locale("en_US.UTF-8"));

  using namespace sequant;
  detail::OpIdRegistrar op_id_registrar;
  mbpt::set_default_convention();
  TensorCanonicalizer::register_instance(
      std::make_shared<DefaultTensorCanonicalizer>());

  auto const calc_info =
      eval::make_calc_info(argv[1], argv[2], argv[3], argc > 4 ? argv[4] : "");

  auto ofs = std::wofstream{argc > 4 ? argv[4] : ""};
  eval::btas::SequantEvalScfBTAS<btas::Tensor<double>>{calc_info}.scf(
      std::wcout);

  return 0;
}