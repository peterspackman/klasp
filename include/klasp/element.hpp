#pragma once
// Element data: symbol, covalent radius, vdW radius, mass (values from occ).

#include <string>
#include <map>

namespace klasp {

struct ElementData {
    int atomic_number;
    float cov_radius;   // Angstrom
    float vdw_radius;   // Angstrom
    float mass;         // amu
};

// Lookup element data by symbol. Returns {0, 0.77, 1.70, 12.0} for unknown.
inline const ElementData& element(const std::string& symbol) {
    static const std::map<std::string, ElementData> table = {
        {"H",  { 1, 0.23f, 1.20f,   1.00794f}},
        {"He", { 2, 1.50f, 1.40f,   4.002602f}},
        {"Li", { 3, 1.28f, 1.82f,   6.941f}},
        {"Be", { 4, 0.96f, 1.53f,   9.012182f}},
        {"B",  { 5, 0.83f, 1.92f,  10.811f}},
        {"C",  { 6, 0.68f, 1.70f,  12.0107f}},
        {"N",  { 7, 0.68f, 1.55f,  14.0067f}},
        {"O",  { 8, 0.68f, 1.52f,  15.9994f}},
        {"F",  { 9, 0.64f, 1.47f,  18.998403f}},
        {"Ne", {10, 1.50f, 1.54f,  20.1797f}},
        {"Na", {11, 1.66f, 2.27f,  22.98977f}},
        {"Mg", {12, 1.41f, 1.73f,  24.305f}},
        {"Al", {13, 1.21f, 1.84f,  26.981538f}},
        {"Si", {14, 1.20f, 2.10f,  28.0855f}},
        {"P",  {15, 1.05f, 1.80f,  30.973761f}},
        {"S",  {16, 1.02f, 1.80f,  32.065f}},
        {"Cl", {17, 0.99f, 1.75f,  35.453f}},
        {"Ar", {18, 1.51f, 1.88f,  39.948f}},
        {"K",  {19, 2.03f, 2.75f,  39.0983f}},
        {"Ca", {20, 1.76f, 2.31f,  40.078f}},
        {"Sc", {21, 1.70f, 2.16f,  44.95591f}},
        {"Ti", {22, 1.60f, 1.87f,  47.867f}},
        {"V",  {23, 1.53f, 1.79f,  50.9415f}},
        {"Cr", {24, 1.39f, 1.89f,  51.9961f}},
        {"Mn", {25, 1.61f, 1.97f,  54.938049f}},
        {"Fe", {26, 1.52f, 1.94f,  55.845f}},
        {"Co", {27, 1.26f, 1.92f,  58.9332f}},
        {"Ni", {28, 1.24f, 1.84f,  58.6934f}},
        {"Cu", {29, 1.32f, 1.86f,  63.546f}},
        {"Zn", {30, 1.22f, 2.10f,  65.409f}},
        {"Br", {35, 1.21f, 1.85f,  79.904f}},
        {"I",  {53, 1.33f, 1.98f, 126.90447f}},
    };
    static const ElementData unknown = {0, 0.77f, 1.70f, 12.0f};
    auto it = table.find(symbol);
    return (it != table.end()) ? it->second : unknown;
}

// Reverse lookup: atomic number -> element symbol. "X" for unknown Z.
inline const char* element_symbol(int z) {
    static const std::map<int, const char*> rev = {
        {1,"H"},{2,"He"},{3,"Li"},{4,"Be"},{5,"B"},{6,"C"},{7,"N"},{8,"O"},{9,"F"},
        {10,"Ne"},{11,"Na"},{12,"Mg"},{13,"Al"},{14,"Si"},{15,"P"},{16,"S"},{17,"Cl"},
        {18,"Ar"},{19,"K"},{20,"Ca"},{21,"Sc"},{22,"Ti"},{23,"V"},{24,"Cr"},{25,"Mn"},
        {26,"Fe"},{27,"Co"},{28,"Ni"},{29,"Cu"},{30,"Zn"},{35,"Br"},{53,"I"},
    };
    auto it = rev.find(z);
    return it != rev.end() ? it->second : "X";
}

}  // namespace klasp
