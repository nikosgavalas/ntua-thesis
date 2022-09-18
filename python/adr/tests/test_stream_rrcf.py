import unittest
import numpy as np

from adr.stream.rrcf import RRCT, Branch


class Test(unittest.TestCase):

    np.random.seed(10)

    def test_0(self):
        '''
        test insert/delete
        '''
        model = RRCT(3, random_state=10)
        point_0 = np.array([[0.76817085, 0.32125623, 0.67318385]])
        point_1 = np.array([[0.3744889, 0.83425859, 0.32878769]])
        point_2 = np.array([[0.99799096, 0.96370693, 0.41436437]])
        point_3 = np.array([[0.73256224, 0.00668613, 0.22025378]])
        point_4 = np.array([[0.50403812, 0.78087025, 0.72392835]])

        model.insert(point_0, 0)
        model.insert(point_1, 1)
        model.insert(point_2, 2)
        model.insert(point_3, 3)
        model.insert(point_4, 4)

        expected = '''└──┐feat: 1, value: 0.29871513009983697, size: 5
   ├──index: (3), point: [0.73256224 0.00668613 0.22025378], size: 1
   └──┐feat: 0, value: 0.9645730807611931, size: 4
      ├──┐feat: 2, value: 0.614883967101137, size: 3
      │  ├──index: (1), point: [0.3744889  0.83425859 0.32878769], size: 1
      │  └──┐feat: 0, value: 0.6147728942937378, size: 2
      │     ├──index: (4), point: [0.50403812 0.78087025 0.72392835], size: 1
      │     └──index: (0), point: [0.76817085 0.32125623 0.67318385], size: 1
      └──index: (2), point: [0.99799096 0.96370693 0.41436437], size: 1
'''

        self.assertEqual(expected, str(model))

        model.remove(0)
        model.remove(1)
        model.remove(2)
        model.remove(3)
        model.remove(4)

    def test_1(self):
        '''
        test scores (simple)
        '''
        model = RRCT(3, random_state=10)
        model.insert(np.array([[1, 1, 1]]), index=0)
        model.insert(np.array([[1, 1, 2]]), index=1)
        model.insert(np.array([[1, 1, 2]]), index=2)
        model.insert(np.array([[2, 1, 1]]), index=3)
        model.insert(np.array([[1, 5, 1]]), index=4)  # anomaly
        expected = [1.0, 1.0, 1.0, 1.0, 4.0]
        for i in range(5):
            self.assertEqual(model.codisp(i), expected[i])

    def test_2(self):
        '''
        test the bounding boxes
        '''
        def check_bboxes(node):
            if isinstance(node, Branch):
                self.assertTrue(np.equal(np.minimum(node.left.bbox[0],
                                                    node.right.bbox[0]),
                                         node.bbox[0]).all())
                self.assertTrue(np.equal(np.maximum(node.left.bbox[1],
                                                    node.right.bbox[1]),
                                         node.bbox[1]).all())
                check_bboxes(node.left)
                check_bboxes(node.right)

        model = RRCT(3, random_state=10)
        for i in range(100):
            model.insert(np.random.uniform(0, 3, size=(1, 3)), index=i)
            check_bboxes(model.root)
        for i in range(50):
            model.remove(index=i)
            check_bboxes(model.root)
        for i in range(50):
            model.insert(np.random.uniform(0, 3, size=(1, 3)), index=i)
            check_bboxes(model.root)
        for i in range(100):
            model.remove(index=i)
            check_bboxes(model.root)

    def test_3(self):
        '''
        test scores (more intense)
        '''
        model = RRCT(4, random_state=10)
        expected = [0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                    1.0, 2.6666666666666665, 1.0, 3.0, 2.4, 2.25,
                    2.1666666666666665, 3.0, 1.0, 4.0, 6.0, 1.0, 1.0,
                    2.7142857142857144, 2.375, 2.111111111111111, 3.0, 1.0,
                    1.8333333333333333, 3.0, 3.0, 1.0, 2.0, 2.0, 1.0,
                    3.8333333333333335, 3.0, 3.0, 2.0, 7.5, 2.0, 2.0, 4.0,
                    3.0, 3.272727272727273, 7.0, 4.428571428571429, 2.5]
        for i in range(50):
            model.insert(np.random.uniform(0, 100, size=(1, 4)), index=i)
            self.assertEqual(model.codisp(i), expected[i])


if __name__ == "__main__":
    unittest.main()
