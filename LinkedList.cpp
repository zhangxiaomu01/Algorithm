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


