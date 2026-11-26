import unittest

from nesy_cvt.cli import build_parser


class CliTests(unittest.TestCase):
    def test_train_parser_accepts_cifar100_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "train",
                "--dataset",
                "cifar100",
                "--data-dir",
                "data",
                "--download",
                "--epochs",
                "1",
            ]
        )

        self.assertEqual(args.dataset, "cifar100")
        self.assertEqual(args.data_dir, "data")
        self.assertTrue(args.download)

    def test_benchmark_parser_accepts_models_and_seeds(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "benchmark",
                "--models",
                "nesy_cvt",
                "resnet18_cifar",
                "--seeds",
                "42",
                "43",
                "--epochs",
                "50",
            ]
        )

        self.assertEqual(args.models, ["nesy_cvt", "resnet18_cifar"])
        self.assertEqual(args.seeds, [42, 43])
        self.assertEqual(args.epochs, 50)


if __name__ == "__main__":
    unittest.main()
