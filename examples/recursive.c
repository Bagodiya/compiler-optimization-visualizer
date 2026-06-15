/* Recursive factorial. Higher levels may turn the tail call into a loop:
 *     compopt show examples/recursive.c
 */

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
