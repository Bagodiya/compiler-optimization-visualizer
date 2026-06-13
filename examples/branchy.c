/* An if/else branch. Compare optimization levels with:
 *     compopt show examples/branchy.c
 */

int max_of_two(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
