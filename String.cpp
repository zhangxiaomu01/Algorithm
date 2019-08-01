//8. String to Integer (atoi)
//https://leetcode.com/problems/string-to-integer-atoi/
/* Not that hard, be careful about the test case */
//Note how we handle the overflow
class Solution {
public:
    int myAtoi(string str) {
        int i = 0,  sign = 1;
        unsigned int ans = 0;
        int len = str.size();
        while(i < len && str[i] == ' ')
            i++;
        
        if(i < len && str[i] == '-'){
            sign = -1;
            i++;
        }else if (i < len && str[i] == '+')
            i++;
        
        while(i < len && (str[i] >= '0' && str[i] <= '9')){
            int backA = ans;
            ans = ans * 10 + (str[i] - '0');
            if(ans < 0 || backA != (ans - (str[i] - '0'))/10 || ans > INT_MAX )
                return sign == 1 ? INT_MAX : INT_MIN;
            i++;
        }
        return ans * sign;
        
    }
};


//3. Longest Substring Without Repeating Characters
//https://leetcode.com/problems/longest-substring-without-repeating-characters/
//The general idea is sliding window. We use hash map to store
//the visited element. Each time when we encounter the duplicate 
//element, we slide the left pointer and until we reach the new 
//duplicate element. Each loop we update the max length and save the 
//visited element to the map
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int len = s.size();
        unordered_map<char, int> map;
        int maxLen = 0;
        int start = 0;
        for(int i = 0; i < len; ++i){
            if(map.find(s[i])!= map.end()){
                //Slide the lowest boundry
                while(map.count(s[i]) > 0) map.erase(s[start++]);
            }
            //Note we shift the start 1 character right
            maxLen = max(maxLen, i - start + 1);
            map[s[i]] = i;
        }
        return maxLen;
    }
};

//The same idea. However, we use int[256] to map the all possible
//characters to potential entry, and can quiaky retrieve the starting
// index
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> dict(256, -1);
        int len = s.size();
        int start = -1, maxLen = 0;
        for(int i = 0; i < len; ++i){
            //We find a duplicate element which has already
            //saved in the dict
            if(dict[s[i]] > start){
                start = dict[s[i]];
            }
            maxLen = max(maxLen, i - start);
            dict[s[i]] = i;
        }
        return maxLen;
    }
};


//387. First Unique Character in a String
//https://leetcode.com/problems/first-unique-character-in-a-string/
/* Test it case by case, nothing special */
class Solution {
public:
    int firstUniqChar(string s) {
        int dict[256];
        fill_n(dict, 256, 0);
        int len = s.size();
        int j = 0;
        for(int i = 0; i < len; ++i){
            int id = s[i];
            dict[id] += 1;
            
            if(i!= j && s[i] == s[j]){
                j++;
                while(j < len && dict[s[j]] > 1)
                    j++;
            }
        }
        
        return j == len ? -1 : j;
    }
};







