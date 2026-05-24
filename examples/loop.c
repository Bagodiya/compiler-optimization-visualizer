/* A simple loop. Compare optimization levels with:
 *     compopt show examples/loop.c
 */

int sum_to_n(int n) {
    int total = 0;
    for (int i = 1; i <= n; i++) {
        total += i;
    }
    return total;
}
