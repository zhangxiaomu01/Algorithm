/**
 * @file Array.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2023-01-15
 * 
 * A quick second pass with the common 'LinkedList' algorithm problems.
 * 
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
 /*
    203. Remove Linked List Elements
    https://leetcode.com/problems/remove-linked-list-elements/
 
    Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.

    Example 1:
    Input: head = [1,2,6,3,4,5,6], val = 6
    Output: [1,2,3,4,5]

    Example 2:
    Input: head = [], val = 1
    Output: []

    Example 3:
    Input: head = [7,7,7,7], val = 7
    Output: []
    
    Constraints:
    The number of nodes in the list is in the range [0, 104].
    1 <= Node.val <= 50
    0 <= val <= 50
 */
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if (head == nullptr) return nullptr;
        ListNode* dummy = new ListNode(-1, head);
        ListNode* ptr = head;
        ListNode* previousNode = dummy;
        while(ptr != nullptr) {
            if (ptr->val == val) {
                ListNode* temp = ptr;
                previousNode->next = ptr->next;
                ptr = ptr->next;
                delete temp;
            } else {
                ptr = ptr->next;
                previousNode = previousNode->next;
            }
        }
        return dummy->next;
    }
};

