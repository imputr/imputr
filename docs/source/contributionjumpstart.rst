Contribution jumpstart
======================

Imputr is an ever-evolving open source library and can always use contributors who want to help build with the community.

Thanks for your interest, let's have a look at we can do!


Create awareness
----------------

The more people we have aboard, the more we can improve the quality of this library!

- **Share our project with the people around you**: you may talk about our project with your co-workers and friends, or post about us
    on your social media accounts. 
- **Write about Imputr**: If you like to write content, you could be of great help to Imputr! 
    You may feature is in a blog article, newsletter or post in which you show how to the library or why you 
    think Imputr is useful to the community.


Contribute to the codebase
--------------------------
- **Report bugs**: inform us of things breaking or improvements you'd like to see by submitting an issues on
  our `issues`_ page. We recommend using our templates for the best way of handling your issue.
- **Smash those bugs**: have a scroll through the bugs in our `issues`_ page. Anything tagged with `bug` and
  `help wanted` is open to whoever wants to implement it.
- **Implement a strategy**: implementing a strategy should be relatively easy to pick up as there are numerous examples. See this `strategy milestone`_ to find some strategy issues you may want to pick up.
- **Implement an enhancement**: look through the GitHub issues for features. Anything tagged with "enhancement"
  and "help wanted" is open to whoever wants to implement it.
- **Write documentation**: it might not seem very glorious, but it is some of the hardest yet most important work to be
  done. Additions to the official Imputr docs, docstrings or even on the web in blog posts or articles are highly
  appreciated!
  
Help the science team
---------------------

The Imputr core science team is constantly trying to learn from Imputr's real-world behavior to best tailor it to the needs of its users.

- **Share your results**: you may share your imputation's accuracy, runtime, memory and (privacy proof) dataset. Tell us what works and does not work for you.
- **Propose research on ideas**: we are constnatly looking for new papers, strategies and heuristics to help us improve our imputations overall. If you have an idea or you happen to find something, please let us know!

.. _issues: https://github.com/imputr/imputr/issues
.. _strategy milestone: https://github.com/imputr/imputr/milestone/1

Getting started
---------------

Ready to contribute code? Here's how to set up `imputr` for local development.


Steps to get started
~~~~~~~~~~~~~~~~~~~~

1. Fork the `imputr` repo on GitHub.
2. Clone your fork locally: ::

    $ git clone git@github.com:your_name_here/imputr.git

3. Ensure poetry_ is installed.


4. Install dependencies and start your virtualenv. Execute the following from within your local repository directory: ::

    $ poetry install

5. Create a branch for local development: ::

    $ git checkout -b name-of-your-bugfix-or-feature

  Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox: ::

    $ poetry run tox

7. Commit your changes and push your branch to GitHub: ::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

.. _poetry: https://python-poetry.org/docs/

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include your tests. We follow a 95% code coverage guideline.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in ``README.md``.
3. The pull request should work for Python 3.7, 3.8, 3.9 and 3.10. Check
   the project's `github actions page`_ and make sure that the tests pass
   for all supported Python versions.

.. _`github actions page`: https://github.com/imputr/imputr/actions