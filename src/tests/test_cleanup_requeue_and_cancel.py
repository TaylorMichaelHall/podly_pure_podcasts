from datetime import UTC, datetime

from app.extensions import db
from app.models import Feed, Post, ProcessingJob
from app.writer.actions.cleanup import cleanup_missing_audio_paths_action
from app.writer.actions.jobs import cancel_queued_jobs_action, mark_cancelled_action


def _create_feed_and_post(app, *, guid="test-guid", audio_path="/tmp/nonexistent.mp3"):
    feed = Feed(
        title="Test Feed",
        description="desc",
        author="author",
        rss_url="https://example.com/feed.xml",
    )
    db.session.add(feed)
    db.session.commit()

    post = Post(
        guid=guid,
        title="Test Episode",
        download_url="https://example.com/ep.mp3",
        feed_id=feed.id,
        whitelisted=True,
        unprocessed_audio_path=audio_path,
    )
    db.session.add(post)
    db.session.commit()
    return feed, post


class TestCleanupRequeuesWhenAudioMissing:
    """Whitelisted posts with missing audio should be re-queued for reprocessing
    regardless of previous job status (except pending/running which are already active)."""

    def test_completed_job_requeued(self, app):
        with app.app_context():
            _, post = _create_feed_and_post(app)
            job = ProcessingJob(
                post_guid=post.guid,
                status="completed",
                completed_at=datetime.now(UTC).replace(tzinfo=None),
            )
            db.session.add(job)
            db.session.commit()

            cleanup_missing_audio_paths_action({})
            db.session.commit()
            db.session.refresh(job)

            assert job.status == "pending"
            assert job.step_name == "Not started"

    def test_failed_job_requeued(self, app):
        with app.app_context():
            _, post = _create_feed_and_post(app, guid="failed-guid")
            job = ProcessingJob(
                post_guid=post.guid,
                status="failed",
                error_message="some error",
                completed_at=datetime.now(UTC).replace(tzinfo=None),
            )
            db.session.add(job)
            db.session.commit()

            cleanup_missing_audio_paths_action({})
            db.session.commit()
            db.session.refresh(job)

            assert job.status == "pending"
            assert job.error_message is None
            assert job.step_name == "Not started"

    def test_pending_job_not_reset(self, app):
        with app.app_context():
            _, post = _create_feed_and_post(app, guid="pending-guid")
            job = ProcessingJob(
                post_guid=post.guid,
                status="pending",
                step_name="Queued",
            )
            db.session.add(job)
            db.session.commit()

            cleanup_missing_audio_paths_action({})
            db.session.commit()
            db.session.refresh(job)

            assert job.status == "pending"
            assert job.step_name == "Queued"


class TestMarkCancelledAction:
    def test_sets_step_name(self, app):
        with app.app_context():
            job = ProcessingJob(post_guid="cancel-test", status="running")
            db.session.add(job)
            db.session.commit()

            mark_cancelled_action({"job_id": job.id, "reason": "Duplicate episode"})
            db.session.commit()
            db.session.refresh(job)

            assert job.status == "cancelled"
            assert job.step_name == "Duplicate episode"
            assert job.error_message == "Duplicate episode"

    def test_default_reason_when_none(self, app):
        with app.app_context():
            job = ProcessingJob(post_guid="cancel-default", status="running")
            db.session.add(job)
            db.session.commit()

            mark_cancelled_action({"job_id": job.id})
            db.session.commit()
            db.session.refresh(job)

            assert job.status == "cancelled"
            assert job.step_name == "Cancelled by user request"
            assert job.error_message == "Cancelled by user request"


class TestCancelQueuedJobsAction:
    def test_cancels_only_pending_jobs(self, app):
        with app.app_context():
            pending = [
                ProcessingJob(post_guid=f"queued-{i}", status="pending")
                for i in range(3)
            ]
            running = ProcessingJob(post_guid="running", status="running")
            db.session.add_all([*pending, running])
            db.session.commit()

            count = cancel_queued_jobs_action({})
            db.session.commit()

            assert count == 3
            for job in pending:
                db.session.refresh(job)
                assert job.status == "cancelled"
                assert job.step_name == "Cancelled by user request"
                assert job.completed_at is not None
            db.session.refresh(running)
            assert running.status == "running"
