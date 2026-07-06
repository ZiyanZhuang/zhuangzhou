from unittest.mock import patch


def test_slash_manager_registers_biodesign_command():
    from zhuangzhou.nlp.slash import manager

    assert "/biodesign" in manager.commands


def test_app_init_does_not_create_prompt_session_eagerly():
    from zhuangzhou.app import ZhuangzhouApp

    with patch("zhuangzhou.app.PromptSession", side_effect=AssertionError("PromptSession should be lazy")):
        app = ZhuangzhouApp()

    assert app.session is None


def test_help_flag_does_not_instantiate_full_app():
    import zhuangzhou.app as app_module

    with patch("sys.argv", ["zhuangzhou", "--help"]):
        with patch.object(app_module, "ZhuangzhouApp", side_effect=AssertionError("App should not be constructed for --help")):
            with patch.object(app_module.ZZInterface, "render_logo"):
                with patch.object(app_module, "_render_help_table"):
                    app_module.main()
