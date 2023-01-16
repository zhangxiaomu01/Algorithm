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

 /*
    707. Design Linked List
    https://leetcode.com/problems/design-linked-list/
 
    Design your implementation of the linked list. You can choose to use a singly or doubly linked list.
    A node in a singly linked list should have two attributes: val and next. val is the value of the current node, and next is a pointer/reference to the next node.
    If you want to use the doubly linked list, you will need one more attribute prev to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

    Implement the MyLinkedList class:

    MyLinkedList() Initializes the MyLinkedList object.
    int get(int index) Get the value of the indexth node in the linked list. If the index is invalid, return -1.
    void addAtHead(int val) Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
    void addAtTail(int val) Append a node of value val as the last element of the linked list.
    void addAtIndex(int index, int val) Add a node of value val before the indexth node in the linked list. If index equals the length of the linked list, the node will be appended to the end of the linked list. If index is greater than the length, the node will not be inserted.
    void deleteAtIndex(int index) Delete the indexth node in the linked list, if the index is valid.
    

    Example 1:

    Input
    ["MyLinkedList", "addAtHead", "addAtTail", "addAtIndex", "get", "deleteAtIndex", "get"]
    [[], [1], [3], [1, 2], [1], [1], [1]]
    Output
    [null, null, null, null, 2, null, 3]

    Explanation
    MyLinkedList myLinkedList = new MyLinkedList();
    myLinkedList.addAtHead(1);
    myLinkedList.addAtTail(3);
    myLinkedList.addAtIndex(1, 2);    // linked list becomes 1->2->3
    myLinkedList.get(1);              // return 2
    myLinkedList.deleteAtIndex(1);    // now the linked list is 1->3
    myLinkedList.get(1);              // return 3
    

    Constraints:

    0 <= index, val <= 1000
    Please do not use the built-in LinkedList library.
    At most 2000 calls will be made to get, addAtHead, addAtTail, addAtIndex and deleteAtIndex.

    * Your MyLinkedList object will be instantiated and called as such:
    * MyLinkedList* obj = new MyLinkedList();
    * int param_1 = obj->get(index);
    * obj->addAtHead(val);
    * obj->addAtTail(val);
    * obj->addAtIndex(index,val);
    * obj->deleteAtIndex(index);
 */
//define the node
struct node{
    int val;
    node* next;
    node(int x) : val(x), next(nullptr){}
};
class MyLinkedList {
private:
    int m_size;
    node* m_lhead;
    node* m_ltail;
    
public:
    /** Initialize your data structure here. */
    MyLinkedList() {
        m_size = 0;
        m_lhead = nullptr;
        m_ltail = nullptr;
    }
    
    void printList(){
        if(!m_lhead) return;
        node* temp = m_lhead;
        while(temp){
            cout << temp->val << " ";
            temp = temp->next;
        }
        cout << endl;
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    int get(int index) {
        if(index < 0 || index >= m_size)
            return -1;
        
        node* temp = m_lhead;
        while(index > 0){
            temp = temp->next;
            index--;
        }
        return temp->val;
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    void addAtHead(int val) {
        node* tempNode = new node(val);
        if(!m_lhead) {
            m_lhead = tempNode;
            m_ltail = tempNode;
            m_size++;
            return;
        }
        tempNode->next = m_lhead;
        m_lhead = tempNode;
        m_size++;
    }
    
    /** Append a node of value val to the last element of the linked list. */
    void addAtTail(int val) {
        if(!m_lhead){
            addAtHead(val);
            return;
        }
        node* tempNode = new node(val);
        m_ltail->next = tempNode;
        m_ltail = tempNode;
        m_size++;
    }
    
    node* getNodeBeforeI(int index){
        int i = index - 1;
        node* temp = m_lhead;
        while(i > 0){
            temp = temp->next;
            i--;
        }
        return temp;
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    void addAtIndex(int index, int val) {
        if(index > m_size) return;
        if(index <= 0) addAtHead(val);
        else if(index == m_size) addAtTail(val);
        else{
            node* temp = getNodeBeforeI(index);
            node* tempNode = new node(val);
            node* tempNextNode = temp->next;
            temp->next = tempNode;
            tempNode->next = tempNextNode;
            m_size++;
        }
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    void deleteAtIndex(int index) {
        if(index < 0 || index >= m_size) return;
        node* temp = getNodeBeforeI(index);
        if(index == 0){
            m_lhead = temp->next;
            m_size--;
            delete temp;
        }
        else{
            if(index == m_size - 1) m_ltail = temp;
            node* pendingDeleteNode = temp->next;
            temp->next = pendingDeleteNode->next;
            m_size--;
            delete pendingDeleteNode;            
        }
    }
};

/*
    206. Reverse Linked List
    https://leetcode.com/problems/reverse-linked-list/
 
    Given the head of a singly linked list, reverse the list, and return the reversed list.

    Example 1:
    Input: head = [1,2,3,4,5]
    Output: [5,4,3,2,1]

    Example 2:
    Input: head = [1,2]
    Output: [2,1]

    Example 3:
    Input: head = []
    Output: []

    Constraints:
    The number of nodes in the list is the range [0, 5000].
    -5000 <= Node.val <= 5000

    Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?
 */
// Iterative
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next) return head;
        ListNode* p = reverseList(head->next);
        ListNode* temp = head->next;
        head->next = nullptr;
        temp->next = head;
        return p;
    }
};

// Recursive
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return head;
        ListNode* reversedList = reverseList(head->next);
        ListNode* ptr = reversedList;
        // The next 2 lines are the brueforce solution, we can easily retrieve the last node of the reversed list
        // with head->next;
        // while(reversedList->next) reversedList = reversedList->next;
        // reverseList->next = head;
        ListNode* temp = head->next;
        temp->next = head;
        head->next = nullptr;
        return ptr;
    }
};

/*
    24. Swap Nodes in Pairs
    https://leetcode.com/problems/swap-nodes-in-pairs/
 
    Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

    Example 1:
    Input: head = [1,2,3,4]
    Output: [2,1,4,3]

    Example 2:
    Input: head = []
    Output: []

    Example 3:
    Input: head = [1]
    Output: [1]
    

    Constraints:
    The number of nodes in the list is in the range [0, 100].
    0 <= Node.val <= 100
 */
// Iterative
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head == nullptr || head -> next == nullptr) return head;
        ListNode* dummy = new ListNode(-1, head);
        ListNode* ptr = head;
        ListNode* next = head->next;
        ListNode* prev = dummy;
        while(next) {
            ptr->next = next->next;
            next->next = ptr;
            swap(ptr, next);
            prev->next = ptr;

            // Move to the next pair
            prev = next;
            ptr = next->next;
            if (ptr == nullptr || ptr->next == nullptr) break;
            next = ptr->next;
        }
        return dummy->next;
    }
};

// Recursive: not from me
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head == NULL || head->next == NULL){
            return head;
        }
        ListNode *p = head->next;
        head->next = swapPairs(head->next->next);
        p->next = head;
        return p;
    }
};
