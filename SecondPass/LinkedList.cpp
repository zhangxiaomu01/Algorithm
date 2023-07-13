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
 
    Given a linked list, swap every two adjacent nodes and return its head. You must solve the 
    problem without modifying the values in the list's nodes (i.e., only nodes themselves may be 
    changed.)

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

/*
    19. Remove Nth Node From End of List
    https://leetcode.com/problems/remove-nth-node-from-end-of-list/
 
    Given the head of a linked list, remove the nth node from the end of the list and return its head.

    Example 1:
    Input: head = [1,2,3,4,5], n = 2
    Output: [1,2,3,5]
    
    Example 2:
    Input: head = [1], n = 1
    Output: []

    Example 3:
    Input: head = [1,2], n = 1
    Output: [1]
    

    Constraints:
    The number of nodes in the list is sz.
    1 <= sz <= 30
    0 <= Node.val <= 100
    1 <= n <= sz

    Follow up: Could you do this in one pass?
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if(!head) return head;
        ListNode dummy(0);
        dummy.next = head;
        ListNode* slow = &dummy, *fast = &dummy;
        //Create a dummy node might be better
        int count = 0;
        
        while(fast){
            if(count > n) slow = slow->next;
            fast = fast->next;
            count++;
        }
        
        ListNode* temp = slow->next;
        slow->next = temp->next;
        delete temp;
        
        return dummy.next;
    }
};

/*
    160. Intersection of Two Linked Lists
    https://leetcode.com/problems/intersection-of-two-linked-lists/
 
    Given the heads of two singly linked-lists headA and headB, return the node at which the two 
    lists intersect. If the two linked lists have no intersection at all, return null.
    For example, the following two linked lists begin to intersect at node c1:


    The test cases are generated such that there are no cycles anywhere in the entire linked structure.
    Note that the linked lists must retain their original structure after the function returns.

    Custom Judge:

    The inputs to the judge are given as follows (your program is not given these inputs):

    intersectVal - The value of the node where the intersection occurs. This is 0 if there is no intersected node.
    listA - The first linked list.
    listB - The second linked list.
    skipA - The number of nodes to skip ahead in listA (starting from the head) to get to the intersected node.
    skipB - The number of nodes to skip ahead in listB (starting from the head) to get to the intersected node.
    The judge will then create the linked structure based on these inputs and pass the two heads, headA and headB to your program. If you correctly return the intersected node, then your solution will be accepted.

    

    Example 1:
    Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
    Output: Intersected at '8'
    Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
    From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
    - Note that the intersected node's value is not 1 because the nodes with value 1 in A and B (2nd node in A and 3rd node in B) are different node references. In other words, they point to two different locations in memory, while the nodes with value 8 in A and B (3rd node in A and 4th node in B) point to the same location in memory.
    
    Example 2:
    Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
    Output: Intersected at '2'
    Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect).
    From the head of A, it reads as [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.
    
    Example 3:
    Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
    Output: No intersection
    Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
    Explanation: The two lists do not intersect, so return null.
    

    Constraints:
    The number of nodes of listA is in the m.
    The number of nodes of listB is in the n.
    1 <= m, n <= 3 * 104
    1 <= Node.val <= 105
    0 <= skipA < m
    0 <= skipB < n
    intersectVal is 0 if listA and listB do not intersect.
    intersectVal == listA[skipA] == listB[skipB] if listA and listB intersect.
    

    Follow up: Could you write a solution that runs in O(m + n) time and use only O(1) memory?
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int lenA = 0, lenB = 0;
        ListNode *pa = headA, *pb = headB;

        while(pa) {
            pa = pa->next;
            lenA++;
        }
        while(pb) {
            pb = pb->next;
            lenB++;
        }

        pa = headA;
        pb = headB;

        // Make sure A is always longer.
        if (lenA < lenB) {
            swap(pa, pb);
            swap(lenA, lenB);
        }

        // Align list a and b together.
        int diff = lenA - lenB;
        while(diff > 0) {
            pa = pa->next;
            diff--;
        }

        while(pa) {
            if(pa == pb) return pa;
            pa = pa->next;
            pb = pb->next; 
        }
        return nullptr;
    }
};

/*
    142. Linked List Cycle II
    https://leetcode.com/problems/linked-list-cycle-ii/
 
    Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.
    There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. Note that pos is not passed as a parameter.
    Do not modify the linked list.

    Example 1:
    Input: head = [3,2,0,-4], pos = 1
    Output: tail connects to node index 1
    Explanation: There is a cycle in the linked list, where tail connects to the second node.

    Example 2:
    Input: head = [1,2], pos = 0
    Output: tail connects to node index 0
    Explanation: There is a cycle in the linked list, where tail connects to the first node.

    Example 3:
    Input: head = [1], pos = -1
    Output: no cycle
    Explanation: There is no cycle in the linked list.
    

    Constraints:
    The number of the nodes in the list is in the range [0, 104].
    -105 <= Node.val <= 105
    pos is -1 or a valid index in the linked-list.
    
    Follow up: Can you solve it using O(1) (i.e. constant) memory?
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        if (fast == nullptr || fast->next == nullptr) return nullptr;

        while(fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
            // Once we detect there is a cycle, the distance from the head to the 
            // entry of the cycle is equal to the distance from the node both
            // pointers are met to the entry of the cycle.
            if (fast == slow) {
                slow = head;
                while(slow != fast) {
                    slow = slow->next;
                    fast = fast->next;
                }
                return slow;
            }
        }
        return nullptr;
    }
};


