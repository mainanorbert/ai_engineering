#include <bits/stdc++.h>
using namespace std;

int main() {
    long long a = 1, b = 2;
    if ((a & 1LL) && (b & 1LL)) {
        cout << "Odd  \n";
    } else {
        cout << "Atlease 1 is even\n";
    }
    return 0;
}
