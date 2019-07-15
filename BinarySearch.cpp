//374. Guess Number Higher or Lower
//https://leetcode.com/problems/guess-number-higher-or-lower/
//Normal binary search
// Forward declaration of guess API.
// @param num, your guess
// @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
int guess(int num);

class Solution {
public:
    int guessNumber(int n) {
        if(n <= 0) return 0;
        int l = 1, r = n;
        while(l <= r){
            int mid = l + (r - l) / 2;
            int response = guess(mid);
            if(response == 0)
                return mid;
            else if (response > 0){
                l = mid + 1;
            }
            else
                r = mid - 1;
        }
        return 0;
    }
};

