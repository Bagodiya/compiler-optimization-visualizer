/* Arithmetic on constants. Higher levels fold this to a single value:
 *     compopt show examples/const_fold.c
 */

int compute(void) {
    int a = 7 * 6;
    int b = a + 100;
    int c = b - 42;
    return c * 2;
}
