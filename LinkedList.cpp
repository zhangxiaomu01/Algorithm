//2. Add Two Numbers
//https://leetcode.com/problems/add-two-numbers/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
//How to structure the code elegantly is critical here
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode pre(0);
        ListNode* p = &pre;
        int carry = 0;
        while(l1 || l2 || carry){
            int res = (l1 ? l1->val : 0) + (l2 ? l2->val : 0) + carry;
            carry = res/10;
            res = res%10;
            p->next = new ListNode(res);
            p = p->next;
            l1 = l1 ? l1->next : l1;
            l2 = l2 ? l2->next : l2;
        }
        return pre.next;
    }
};

//445. Add Two Numbers II
//https://leetcode.com/problems/add-two-numbers-ii/
//Reverse listNode, and then sum them together
//exact the same as Add Two Numbers I
class Solution {
private:
    ListNode* ReverseList(ListNode* l){
        if(!l || !l->next) return l;
        ListNode* p = ReverseList(l->next);
        l->next->next = l;
        l->next = nullptr;
        return p;
    }
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        l1 = ReverseList(l1);
        l2 = ReverseList(l2);
        ListNode head(0);
        ListNode* ptr = &head;
        int carry = 0;
        while(l1 || l2 || carry){
            int sum = (l1 ? l1->val : 0) + (l2 ? l2->val : 0) + carry;
            carry = sum / 10;
            int newVal = sum % 10;
            ptr->next = new ListNode(newVal);
            ptr = ptr->next;
            l1 = (l1 ? l1->next : nullptr);
            l2 = (l2 ? l2->next : nullptr);
        }
        ptr = ReverseList(head.next);
        return ptr;
    }
};



//206. Reverse Linked List
//https://leetcode.com/problems/reverse-linked-list/
//Both versions are tricky!
//Note *p always points to the last node of the original array
//At that time, current head still points to the last element 
//of reverse(head->naxt)
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next) return head;
        ListNode* p = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return p;
    }
};

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr, *cur = head;
        while(cur){
            ListNode* temp = cur->next;
            cur->next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }
};


//21. Merge Two Sorted Lists
//https://leetcode.com/problems/merge-two-sorted-lists/
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 && !l2) return l1;
        else if (l2 && !l1) return l2;
        else if(!l1 && !l2) return nullptr;
        
        ListNode* dummy = new ListNode(0);
        ListNode* pre = dummy;
            
        while(l1 && l2){
            if(l1->val < l2->val){
                pre->next = l1;
                pre = pre->next;
                l1 = l1->next;
            }else{
                pre->next = l2;
                pre = pre->next;
                l2 = l2->next;
            }
        }
        if(l1) pre->next = l1;
        else if(l2) pre->next = l2;
        ListNode* ptr = dummy->next;
        delete dummy;
        return ptr;
    }
};


class Solution {
private:
    ListNode* helper(ListNode* l1, ListNode* l2){
        if(!l1 && !l2) return nullptr;
        if(!l1) return l2;
        if(!l2) return l1;
        
        if(l1->val <= l2->val){
            ListNode* node = helper(l1->next, l2);
            l1->next = node;
            return l1;
        }else{
            ListNode* node = helper(l1, l2->next);
            l2->next = node;
            return l2;
        }
    }
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        return helper(l1, l2);
    }
};


//142. Linked List Cycle II
//https://leetcode.com/problems/linked-list-cycle-ii/
//Cycle detection. Same as problem 287
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(!head) return head;
        ListNode* fast = head, *slow = head;
        do{
            fast = fast ? fast->next : nullptr;
            fast = fast ? fast->next : nullptr;
            slow = slow->next;
        }while(fast != slow);
        if(!fast) return nullptr;
        
        slow = head;
        while(slow != fast){
            slow = slow->next;
            fast = fast->next;
        }
        return fast;      
    }
};


//234. Palindrome Linked List
//https://leetcode.com/problems/palindrome-linked-list/
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        vector<int> dict;
        ListNode* ptr = head;
        while(ptr){
            dict.push_back(ptr->val);
            ptr = ptr->next;
        }
        int i = 0, j = dict.size() - 1;
        while(i < j){
            if(dict[i] != dict[j]) return false;
            i++;
            j--;
        }
        return true;
    }
};

/* Optimized version O(n) time, O(1) space */
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        ListNode* fast = head, *slow = head;
        if(!head || !head->next) return true;
        //find the midpoint of the list
        /*
        For list with even elements
        1 2 2 1
          ^
        For list with odd elements
        1 2 3 2 1
            ^
        */
        while(fast && fast->next){
            if(!fast->next->next)
                break;
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode* nHead = slow->next;
        //Reverse the right half of the list
        ListNode* pre = nHead;
        ListNode* cur = nHead ? nHead->next : nullptr;
        //break the first node to prevent loops
        if(cur) pre->next = nullptr;
        while(cur){
            ListNode* temp = cur->next;
            cur->next = pre;
            //pre->next = nullptr;
            pre = cur;
            cur = temp;
        }
        
        //Now pre becomes the head of the reversed list
        //Link the list now
        nHead = pre;
        cur = head;
        while(pre){
            if(pre->val != cur->val)
                return false;
            pre = pre->next;
            cur = cur->next;
        }
        //You can reverse the array back if you want
        return true;
        
    }
};


//328. Odd Even Linked List
//https://leetcode.com/problems/odd-even-linked-list/
/* General idea: split the list based on even and odd number. Merge them together */
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if(!head || !head->next) return head;
        ListNode oNode(1), eNode(2);
        oNode.next = head, eNode.next = head->next;
        ListNode* oddPtr = head, *evenPtr = head->next;
        while(oddPtr && evenPtr){
            oddPtr->next = evenPtr->next;
            oddPtr = oddPtr->next;
            swap(oddPtr, evenPtr);
        }
        oddPtr = &oNode, evenPtr = &eNode;
        while(oddPtr->next){
            oddPtr = oddPtr->next;
        }
        
        oddPtr->next = evenPtr->next;
        return oNode.next;
    }
};

//stack solution
//Using stack, exactly the same idea. Without changing the original 
//list
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        vector<int> nums1, nums2;
        while(l1) {
            nums1.push_back(l1->val);
            l1 = l1->next;
        }
        while(l2) {
            nums2.push_back(l2->val);
            l2 = l2->next;
        }

        int m = nums1.size(), n = nums2.size();
        int sum = 0, carry = 0;

        ListNode *head = nullptr, *p = nullptr;

        for(int i = m - 1, j = n - 1; i >= 0 || j >= 0 || carry > 0; i--, j--) {
            sum = carry;
            if(i >= 0) 
                sum += nums1[i];

            if(j >= 0)
                sum += nums2[j];

            carry = sum / 10;

            p = new ListNode(sum%10);
            p->next = head;
            head = p;
        }

        return head;
    }
};

//1171. Remove Zero Sum Consecutive Nodes from Linked List
//https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/
//Prefix sum is the key!
class Solution {
private:
    //[s, e)
    void releaseMemory(ListNode* s, ListNode* e, int preSum, unordered_map<int, ListNode*>& uMap){
        while(s != e){
            preSum += s->val;
            //When s->next == e, we abort. Since we cannot delete original
            //preSum
            if(s && s->next != e){
                uMap.erase(preSum);
            }
            
            ListNode* temp = s;
            s = s->next;
            //It's weird that when I delete the node, there will be runtime
            //error, however, in play ground, it looks good.
            //It seems that when the test case need to delete the leftmost 
            //value, then error occurs!
            delete temp;
        }
    }
public:
    ListNode* removeZeroSumSublists(ListNode* head) {
        if(!head) return head;
        //uMap for prefix sum
        unordered_map<int, ListNode*> uMap;
        //Since all node with value 0 will be removed, we need to 
        //initialize dummy to be 1001 here
        ListNode dummy(10001);
        dummy.next = head;
        ListNode* node = &dummy;
        int prefixSum = 0;
        while(node != nullptr){
            prefixSum += node->val;
            //Which means we have add a value and cancel the sum of [i..j]
            //which means sum of [i..j] to be 0
            if(node->val == 0 || uMap.count(prefixSum) > 0){
                //cout << prefixSum <<endl;
                ListNode* tempNode = uMap[prefixSum];
                ListNode* deleteStartNode = tempNode->next;
                tempNode->next = node->next;
                releaseMemory(deleteStartNode, node->next, prefixSum, uMap);
                //note here node has already been deleted, we need to reset
                //node to tempNode->next
                node = tempNode->next;
                continue;
            }
            uMap[prefixSum] = node;
            node = node->next;
        }
        
        return dummy.next;
    }
};


//24. Swap Nodes in Pairs
//https://leetcode.com/problems/swap-nodes-in-pairs/
//Recursive solution!
class Solution {
    ListNode* helper(ListNode* node){
        if(!node || !node->next) return node;
        ListNode* tempNode = node->next->next;
        ListNode* newHead = node->next;
        newHead->next = node;
        node->next = helper(tempNode);
        return newHead;
        
    }
public:
    ListNode* swapPairs(ListNode* head) {
        return helper(head);
    }
};

//Iterative Version
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* p = dummy;
        while(p->next&&p->next->next){
            ListNode* p1 = p->next;
            ListNode* p2 = p->next->next;
            p->next = p2;
            p1->next = p2->next;
            p2->next = p1;
            p = p1;
        }
        return dummy->next;
        
    }
};



//708. Insert into a Cyclic Sorted List
//https://leetcode.com/problems/insert-into-a-cyclic-sorted-list/
//How to optimize code is the key!
class Solution {
public:
    Node* insert(Node* head, int val){
      if(!head){
        head = new Node(val);
        head->next = head;
        return head;
      }

      Node* fPtr = head, *sPtr = head;
      sPtr = head->next;
      while(sPtr != head){
        if(fPtr->val <= val && sPtr->val >= val){
          break;
        }
        if(fPtr->val > sPtr->val && (fPtr->val <= val || sPtr->val >= val)){
          break;
        }
          fPtr = sPtr;
          sPtr = sPtr->next;
      }

      Node* tempNode = new Node(val);
      fPtr->next = tempNode;
      tempNode->next = sPtr;
      return head;

    }
};



//23. Merge k Sorted Lists
//https://leetcode.com/problems/merge-k-sorted-lists/
/* priority_queue solution! */
class Solution {
public:
   
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int len_l = lists.size();
        ListNode head(0);
        ListNode *h = &head;
        auto comp = [](ListNode* l1, ListNode* l2){
            return l1->val> l2->val;
        };
        priority_queue<ListNode*, std::vector<ListNode*>, decltype(comp)> q(comp);
        
        for(int i = 0; i < len_l; i++){
            if(lists[i])
                q.push(lists[i]);
        }

        while(q.empty() == false){
            h->next = q.top();
            q.pop();
            h = h->next;
            if(h&&h->next != NULL)
                q.push(h->next); 
        }
        
        return head.next;
    }
};


//Merge sort step. Very celever!
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2){
        ListNode head(0);
        ListNode* h = &head;
        while(l1&&l2){
            if(l1->val < l2->val){
                h->next = l1;
                h=h->next;
                l1= l1->next;
            }
            else{
                h->next = l2;
                h = h->next;
                l2=l2->next;
            }
        }
        if(l1)
        {
            h->next = l1;
        }
        if(l2)
        {
            h->next = l2;
        }
        
        return head.next;
    }
    
    
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int len_l = lists.size();
        if(len_l == 0)
            return nullptr;
        
        int step = 1;
        while(step < len_l){
            for(int i = 0; i + step < len_l; i = i + step*2)
                {
                    lists[i] = mergeTwoLists(lists[i], lists[i+step]);
                }
            step = step*2;
        }
        return lists[0];
    }
};


//19. Remove Nth Node From End of List
//https://leetcode.com/problems/remove-nth-node-from-end-of-list/
//Follow up: with one pass. Two pointers!
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if(!head || !(head->next))
            return nullptr;
        
        ListNode *fast = head, *slow =head;
        int count = 0;
        
        while(fast->next != nullptr)
        {
            fast = fast->next;
            count ++;
            if(count> n)
            {
                slow = slow->next;
            }
        }
        if(count + 1 == n)
        {
            head = head-> next;
            delete slow;
            return head;
        }
        
        ListNode *temp = slow->next;
        if(temp == nullptr)
            return head;
        slow->next = temp->next;
        delete temp;
        
        return head;
        
    }
};


//138. Copy List with Random Pointer
//https://leetcode.com/problems/copy-list-with-random-pointer/
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node() {}

    Node(int _val, Node* _next, Node* _random) {
        val = _val;
        next = _next;
        random = _random;
    }
};
*/
//Build a map with previous node and new node
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(!head) return nullptr;
        unordered_map<Node*, Node*> uMap;
        Node* ptr = head;
        Node dummy(0, nullptr, nullptr);
        Node* ptr2 = &dummy;
        while(ptr){
            ptr2->next = new Node(ptr->val, nullptr, nullptr);
            ptr2 = ptr2->next;
            uMap[ptr] = ptr2;
            ptr = ptr->next;
        }
        
        ptr = head;
        ptr2 = dummy.next;
        while(ptr2){
            ptr2->random = uMap[ptr->random];
            ptr2 = ptr2->next;
            ptr = ptr->next;
        }
        return dummy.next;
    }
};


//Without map. Hard to get the insight and get it right!
class Solution {
public:
    RandomListNode *copyRandomList(RandomListNode *head) {
        if(head == NULL) return NULL;
        RandomListNode* ptr = head;
        //First pass, we interleave new node with old node
        //New node is a copy of old node
        while(ptr!=NULL){
            RandomListNode* node = new RandomListNode(ptr->label);
            node->next = ptr->next;
            ptr->next = node;
            ptr = ptr->next->next;
        }
        //Second Pass: set random pointers
        ptr = head;
        while(ptr!=NULL){
            ptr->next->random = ptr->random == NULL ? NULL:ptr->random->next;
            ptr = ptr->next->next;
        }
        //Last pass: separate two lists
        RandomListNode* oldList = head;
        RandomListNode* newList = head->next;
        RandomListNode* newHead = head->next;
        while(oldList!=NULL){
            oldList->next = newList->next;
            newList->next = oldList->next == NULL? NULL: oldList->next->next;
            oldList = oldList->next;
            newList = newList->next;
        }
        return newHead;
        
    }
};



//369. Plus One Linked List
//https://leetcode.com/problems/plus-one-linked-list/
class Solution {
private:
    int helper(ListNode* node){
        if(!node) return 0;
        
        int carry = helper(node->next);
        int val = node->val + carry;
        if(node->next == nullptr){
            val += 1;
        }
        node->val = val % 10;
        carry = val / 10;
        return carry;
    }
public:
    ListNode* plusOne(ListNode* head) {
        ListNode* addOn = new ListNode(0);
        addOn->next = head;
        helper(addOn);
        if(addOn->val > 0) return addOn;
        delete addOn;
        return head;
    }
};


