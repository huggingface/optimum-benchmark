
# How to contribute to Optimum-Benchmark?

`optimum-benchmark` is an open source project, so all contributions and suggestions are welcome.

You can contribute in many different ways: giving ideas, answering questions, reporting bugs, proposing enhancements, improving the documentation, fixing bugs,...

Many thanks in advance to every contributor.

## How to work on an open Issue?

You have the list of open Issues at: <https://github.com/huggingface/optimum-benchmark/issues>

If you would like to work on any of the open Issues:

1. Make sure it is not already assigned to someone else. You have the assignee (if any) on the top of the right column of the Issue page. If it is not assigned, you can assign it to yourself by clicking on the "Assign yourself" button, or by leaving a comment on the Issue page.

2. Create a Pull Request.

## How to create a Pull Request?

1. Fork the [repository](https://github.com/huggingface/optimum-benchmark) by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

	```bash
	git clone https://github.com/<Your Github Username>/optimum-benchmark.git
	cd optimum-benchmark
	git remote add upstream https://github.com/huggingface/optimum-benchmark.git
	```

3. Create a new branch to hold your development changes:

	```bash
	git checkout -b name-of-your-branch
	```
	
	**do not** work on the `main` branch.

4. Set up a development environment by running the following command in a virtual environment:

	```bash
	pip install -e .[quality,testing]
	```

5. Develop the features or fix the bug you want to work on.

6. Depending on the feature you're working on and your development environment, you can run tests locally in an isolated docker container using the [makefile](Makefile). For example, to test the CLI with CPU device and PyTorch backend, you can run the following commands:

	```bash
	make install_cli_cpu_pytorch
	make test_cli_cpu_pytorch
	```

	For a better development experience, we recommend using isolated docker containers to run tests:
	
	```bash
	make build_cpu_image
	make run_cpu_container
	make install_cli_cpu_pytorch
	make test_cli_cpu_pytorch
	```

	You can find more information about the available make commands in the [Makefile](Makefile).

7. Make sure your code is properly formatted and linted by running:

	```bash
	make style
	```

8. Once you're happy with your changes, add the changed files using `git add` and make a commit with `git commit` to record your changes locally:

	```bash
	git add modified_file.py
	git commit
	```

	It is a good idea to sync your copy of the code with the original repository regularly. This way you can quickly account for changes:

	```bash
	git fetch upstream
	git rebase upstream/main
	```

	Push the changes to your account using:

	```bash
	git push -u origin name-of-your-branch
	```

9. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send your to the project maintainers for review.
