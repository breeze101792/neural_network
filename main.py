#!/usr/bin/env python3
from gi.repository import Gtk, Gdk
from gi.repository.GdkPixbuf import Pixbuf
import matplotlib.pyplot as plt
import threading
from perceptron import *
import numpy as np
from data_proc import data_proc as dp
from ploting import paper
from matplotlib import colors

# color = ["r","b","g","c","m","y","k","w"]

class myDict(dict):

    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value


UI_INFO = """
<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <!--menu action='FileNew'>
        <menuitem action='FileNewStandard' />
        <menuitem action='FileNewFoo' />
        <menuitem action='FileNewGoo' />
      </menu-->
      <menuitem action='FileOpen' />
      <menuitem action='FileSave' />
      <separator />
      <menuitem action='FileQuit' />
    </menu>
    <menu action='AboutMenu'>
      <menuitem action='About'/>
    </menu>
  </menubar>
  <toolbar name='ToolBar'>
    <!--toolitem action='FileNewStandard' /-->
    <toolitem action='FileOpen' />
    <!--toolitem action='FileSave' /-->
    <toolitem action='Run' />
    <toolitem action='FileQuit' />
  </toolbar>
  <!--popup name='PopupMenu'>
    <menuitem action='EditCopy' />
    <menuitem action='EditPaste' />
    <menuitem action='EditSomething' />
  </popup-->
</ui>
"""
class info:
    class traning_data:
        def __init__(self):
            self.Iteration_times = 0
            self.Error_rate = 0
            self.Data_set_size = 0
        def reset(self):
            self.Iteration_times = 0
            self.Error_rate = 0
            self.Data_set_size = 0
    class testing_data:
        def __init__(self):
            self.Error_rate = 0
            self.Data_set_size = 0
        def reset(self):
            self.Error_rate = 0
            self.Data_set_size = 0
    def __init__(self):
        self.traning = self.traning_data()
        self.testing = self.testing_data()
    def reset(self):
        self.traning.reset()
        self.testing.reset()

class nNetwork(Gtk.Window):
    def __init__(self):
        self.nnetwork = None
        self.dimension = 2
        self.train_mode = True
        #self.data = []
        self.weights = []
        self.class_table = {}
        self.find_best = False
        #self.class_num
        self.dataset = dp()
        self.training_set = []
        self.testing_set = []
        self.traning_trainsformed_data = []
        self.testing_trainsformed_data = []
        #log info
        self.nninfo = info()

        #wait for ui setup
        Gtk.Window.__init__(self, title="Neural Network")
        self.set_default_size(800, 700)

        action_group = Gtk.ActionGroup("my_actions")

        #menu item
        self.add_file_menu_actions(action_group)
        self.add_about_menu_actions(action_group)

        #create ui manager
        uimanager = self.create_ui_manager()
        uimanager.insert_action_group(action_group)

        menubar = uimanager.get_widget("/MenuBar")

        main_ui = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(main_ui)
        main_ui.pack_start(menubar, False, False, 0)

        toolbar = uimanager.get_widget("/ToolBar")
        main_ui.pack_start(toolbar, False, False, 0)

        # body pannel settings
        body_panel = Gtk.Table(2, 3, True)
        main_ui.pack_start(body_panel, True, True, 0)

        #for testing/*********************************************************/
        settings_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing = 10)
        body_panel.attach(settings_panel, 0, 1, 0, 2, xpadding=8, ypadding=8)

        settings_lab = Gtk.Label("Settings", xalign=0)
        settings_panel.pack_start(settings_lab, False, False, 0)

        # data settings *******************************************************/
        data_settings_lab = Gtk.Label("Data Settings", xalign=0)
        settings_panel.pack_start(data_settings_lab, False, False, 0)

        # traning_testing_rate_group
        traning_testing_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(traning_testing_rate_group, False, False, 0)
        traning_testing_rate_lab = Gtk.Label("Traning & Tesgting\ndata rate(%):", xalign=0)
        traning_testing_rate_group.attach(traning_testing_rate_lab, 0, 1, 0, 1)
        traning_testing_rate_adj = Gtk.Adjustment(60, 0, 100, 5, 0, 0)
        self.traning_testing_rate_sb = Gtk.SpinButton()
        self.traning_testing_rate_sb.set_alignment(xalign=1)
        self.traning_testing_rate_sb.set_adjustment(traning_testing_rate_adj)
        traning_testing_rate_group.attach(self.traning_testing_rate_sb, 1, 2, 0, 1)
        self.traning_testing_rate_sb.set_value(66)

        # action buttom
        data_action_group = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        settings_panel.pack_start(data_action_group, False, False, 0)

        generate_button = Gtk.Button(label = "generate")
        generate_button.connect("clicked", self.on_clicked_generate)
        data_action_group.pack_start(generate_button, False, False, 0)

        # test_button = Gtk.Button(label = "Test")
        # test_button.connect("clicked", self.on_clicked_test)
        # data_action_group.pack_start(test_button, False, False, 0)

        # network settings*****************************************************
        nn_settings_lab = Gtk.Label("Network Settings", xalign=0)
        settings_panel.pack_start(nn_settings_lab, False, False, 0)

        # learning rate
        learning_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(learning_rate_group, False, False, 0)
        learning_rate_lab = Gtk.Label("Learning rate(0.1):", xalign=0)
        learning_rate_group.attach(learning_rate_lab, 0, 1, 0, 1)
        learning_rate_adj = Gtk.Adjustment(2, 0, 100, 1, 10, 0)
        self.learning_rate_sb = Gtk.SpinButton()
        self.learning_rate_sb.set_alignment(xalign=1)
        self.learning_rate_sb.set_adjustment(learning_rate_adj)
        learning_rate_group.attach(self.learning_rate_sb, 1, 2, 0, 1)
        self.learning_rate_sb.set_value(6)

        # traning times
        training_times_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(training_times_group, False, False, 0)
        training_times_lab = Gtk.Label("Traning times:", xalign=0)
        training_times_group.attach(training_times_lab, 0, 1, 0, 1)
        training_times_adj = Gtk.Adjustment(50, 0, 10000, 20, 0, 0)
        self.training_times_sb = Gtk.SpinButton()
        self.training_times_sb.set_alignment(xalign=1)
        self.training_times_sb.set_adjustment(training_times_adj)
        training_times_group.attach(self.training_times_sb, 1, 2, 0, 1)
        self.training_times_sb.set_value(100)

        # training_err_rate_group
        training_err_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(training_err_rate_group, False, False, 0)
        training_err_rate_lab = Gtk.Label("Traning error rate(%):", xalign=0)
        training_err_rate_group.attach(training_err_rate_lab, 0, 1, 0, 1)
        training_err_rate_adj = Gtk.Adjustment(10, 0, 100, 5, 0, 0)
        self.training_err_rate_sb = Gtk.SpinButton()
        self.training_err_rate_sb.set_alignment(xalign=1)
        self.training_err_rate_sb.set_adjustment(training_err_rate_adj)
        training_err_rate_group.attach(self.training_err_rate_sb, 1, 2, 0, 1)
        self.training_err_rate_sb.set_value(5)



        # mlp structure
        mlp_structure_group = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        settings_panel.pack_start(mlp_structure_group, False, False, 0)
        mlp_structure_lab = Gtk.Label("Specify mlp structure(3,5,1...)", xalign = 0)
        mlp_structure_group.pack_start(mlp_structure_lab, True, True, 0)
        self.mlp_structure_ety = Gtk.Entry()
        self.mlp_structure_ety.set_alignment(xalign=1)
        self.mlp_structure_ety.set_text("2,1")
        mlp_structure_group.pack_start(self.mlp_structure_ety, True, True, 0)


        # action buttom
        action_group = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        settings_panel.pack_start(action_group, False, False, 0)

        tarin_button = Gtk.Button(label = "Train")
        tarin_button.connect("clicked", self.on_clicked_train)
        action_group.pack_start(tarin_button, False, False, 0)

        test_button = Gtk.Button(label = "Test")
        test_button.connect("clicked", self.on_clicked_test)
        action_group.pack_start(test_button, False, False, 0)

        # draw_button = Gtk.Button(label = "draw")
        # draw_button.connect("clicked", self.draw_2dfig())
        # action_group.pack_start(draw_button, False, False, 0)

        # draw_button = Gtk.Button(label = "Draw")
        # draw_button.connect("clicked", self.on_clicked_draw)
        # action_group.pack_start(draw_button, False, False, 0)

        # /*******************************************************************/

        #info*****************************************************************/
        info_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing = 10)
        body_panel.attach(info_panel, 1, 2, 0, 2)

        info_lab = Gtk.Label("Information", xalign=0)
        info_panel.pack_start(info_lab, False, False, 0)

        #data info
        dataset_info_lab = Gtk.Label("#Data Info", xalign=0)
        info_panel.pack_start(dataset_info_lab, False, False, 0)
        dataset_info_group = Gtk.Table(1, 2, False)
        info_panel.pack_start(dataset_info_group, False, False, 0)
        self.dataset_info_title_lab = Gtk.Label("FileName:\nData set size:\nDimension:\nClassification number:", xalign=0, yalign=0)
        dataset_info_group.attach(self.dataset_info_title_lab, 0,1,0,1)
        self.dataset_info_msg_lab = Gtk.Label("", xalign=0, yalign=0)
        dataset_info_group.attach(self.dataset_info_msg_lab, 1,2,0,1)

        #taning log
        traning_log_lab = Gtk.Label("#Traning Log", xalign=0)
        info_panel.pack_start(traning_log_lab, False, False, 0)
        traning_log_group = Gtk.Table(1, 2, False)
        info_panel.pack_start(traning_log_group, False, False, 0)
        self.traning_log_title_lab = Gtk.Label("Accuracy rate:\nTraning set size:\nIteration times:", xalign=0, yalign=0)
        traning_log_group.attach(self.traning_log_title_lab, 0,1,0,1)
        self.traning_log_msg_lab = Gtk.Label("", xalign=0, yalign=0)
        traning_log_group.attach(self.traning_log_msg_lab, 1,2,0,1)

        #testing log
        testing_log_lab = Gtk.Label("#Testing Log", xalign=0, yalign=0)
        info_panel.pack_start(testing_log_lab, False, False, 0)
        testing_log_group = Gtk.Table(1, 2, False)
        info_panel.pack_start(testing_log_group, False, False, 0)
        self.testing_title_log_lab = Gtk.Label("Accuracy rate:\nTesting set size:", xalign=0, yalign=0)
        testing_log_group.attach(self.testing_title_log_lab, 0,1,0,1)
        self.testing_log_msg_lab = Gtk.Label("", xalign=0, yalign=0)
        testing_log_group.attach(self.testing_log_msg_lab, 1,2,0,1)

        ori_draw_panel = Gtk.Box(10, 2, True)
        info_panel.pack_start(ori_draw_panel, True, True, 0)

        self.ori_paper = paper(title="After normalization(data set)")
        ori_draw_panel.pack_start(self.ori_paper.canvas, True, True, 0)

        # *********************************************************************/
        #drawing
        traning_draw_draw_panel = Gtk.Box(10, 2, True)
        body_panel.attach(traning_draw_draw_panel, 2, 3, 0, 1)

        self.traning_draw_paper = paper(title="Traning set")
        traning_draw_draw_panel.pack_start(self.traning_draw_paper.canvas, True, True, 0)
        self.traning_draw_paper.resetpaper()



        testing_draw_draw_panel = Gtk.Box(10, 2, True)
        body_panel.attach(testing_draw_draw_panel, 2, 3, 1, 2)

        self.testing_draw_paper = paper(title="Testing set")
        testing_draw_draw_panel.pack_start(self.testing_draw_paper.canvas, True, True, 0)
        self.testing_draw_paper.resetpaper()

        #post init
        self.log_refresh()

        # self.classifier = None
    def add_file_menu_actions(self, action_group):
        action_filemenu = Gtk.Action("FileMenu", "File", None, None)
        action_group.add_action(action_filemenu)

        action_filenewmenu = Gtk.Action("FileNew", None, None, Gtk.STOCK_NEW)
        action_group.add_action(action_filenewmenu)

        action_new = Gtk.Action("FileNewStandard", "_New",
            "Create a new file", Gtk.STOCK_NEW)
        action_new.connect("activate", self.on_menu_file_new_generic)
        action_group.add_action_with_accel(action_new, None)

        action_group.add_actions([
            ("FileNewFoo", None, "New Foo", None, "Create new foo",
             self.on_menu_file_new_generic),
            ("FileNewGoo", None, "_New Goo", None, "Create new goo",
             self.on_menu_file_new_generic),
        ])

        action_fileopen = Gtk.Action("FileOpen", "Open", None, Gtk.STOCK_OPEN)
        action_fileopen.connect("activate", self.on_menu_file_open)
        action_group.add_action(action_fileopen)

        action_filesave = Gtk.Action("FileSave", "Save", None, Gtk.STOCK_SAVE)
        action_group.add_action(action_filesave)

        action_run = Gtk.Action("Run", "Run", None, Gtk.STOCK_MEDIA_PLAY)
        action_run.connect("activate", self.on_clicked_run)
        action_group.add_action(action_run)

        action_filequit = Gtk.Action("FileQuit", None, None, Gtk.STOCK_QUIT)
        action_filequit.connect("activate", self.on_menu_file_quit)
        action_group.add_action(action_filequit)

    def add_about_menu_actions(self, action_group):
        action_aboutmenu = action_group.add_action(Gtk.Action("AboutMenu", "About", None,
            None))
        action_aboutmenu = Gtk.Action("About", "About", None, self.on_menu_others)
        action_group.add_action(action_aboutmenu)

    def create_ui_manager(self):
        uimanager = Gtk.UIManager()

        # Throws exception if something went wrong
        uimanager.add_ui_from_string(UI_INFO)

        # Add the accelerator group to the toplevel window
        accelgroup = uimanager.get_accel_group()
        self.add_accel_group(accelgroup)
        return uimanager

    def on_menu_file_new_generic(self, widget):
        print("A File|New menu item was selected.")

    def on_menu_file_quit(self, widget):
        Gtk.main_quit()

    def on_menu_file_open(self, widget):
        dialog = Gtk.FileChooserDialog("Please choose a file", self,
            Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        self.add_filters(dialog)

        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            #reset
            self.nninfo.reset()
            self.log_refresh()

            print("Open clicked")
            print("File selected: " + dialog.get_filename())
            print(self.traning_testing_rate_sb.get_value() / 100)
            self.dataset.set_file_name(dialog.get_filename())
            self.dataset.open_file()
            # if self.dataset.get_data_size() > 100:
            #     print("tt rate\t", self.traning_testing_rate_sb.get_value())
            #     self.training_set = self.dataset.get_data(self.traning_testing_rate_sb.get_value() / 100)
            #     self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
            #     self.testing_set = self.dataset.get_data(1 - self.traning_testing_rate_sb.get_value() / 100)
            #     self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])
            # else:
            #     self.training_set = self.dataset.get_data(1)
            #     self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
            #     self.testing_set = self.dataset.get_data(1)
            #     self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])

            #log info
            file_name = dialog.get_filename().split('/')[-1]


            if self.dataset.get_data_size() > 100:
                print("tt rate\t", self.traning_testing_rate_sb.get_value())
                self.training_set = self.dataset.get_data(self.traning_testing_rate_sb.get_value() / 100)
                self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
                self.testing_set = self.dataset.get_data(1 - self.traning_testing_rate_sb.get_value() / 100)
                self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])
            else:
                self.training_set = self.dataset.get_data(1)
                self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
                self.testing_set = self.dataset.get_data(1)
                self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])
            self.nninfo.traning.Data_set_size = len(self.training_set[1])
            self.nninfo.testing.Data_set_size = len(self.testing_set[1])

            if self.dataset.get_data_dimension() == 2:
                self.ori_paper.resetpaper()
                self.ori_paper.draw_2d_point(self.dataset.get_data(1, is_random = True),self.dataset.get_class_middle())
                self.ori_paper.expend_lim()
                self.ori_paper.draw()
            else:
                self.ori_paper.resetpaper()
                self.ori_paper.draw()

            self.dataset_info_msg_lab.set_text((file_name[0:18] + "..." if len(file_name) > 20 else file_name) + " \n" + self.dataset.get_data_size().__str__() + " \n" + self.dataset.get_data_dimension().__str__() + " \n" + self.dataset.get_data_classification_num().__str__() + " ")

            self.traning_draw_paper.resetpaper()
            self.testing_draw_paper.resetpaper()
        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")
        #print(self.data, ", ", self.class_table, ", ", tmp_table)
        dialog.destroy()
    def add_filters(self, dialog):
        filter_text = Gtk.FileFilter()
        filter_text.set_name("Text files")
        filter_text.add_mime_type("text/plain")
        dialog.add_filter(filter_text)

        filter_py = Gtk.FileFilter()
        filter_py.set_name("Python files")
        filter_py.add_mime_type("text/x-python")
        dialog.add_filter(filter_py)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)
    def on_find_best_activated(self, switch, gparam):

        if switch.get_active():
            self.find_best = True
        else:
            self.find_best = False

    def log_refresh(self, widget = None, pannel="ALL"):
        tmp_str = ""
        if pannel == "ALL":
            tmp_str = format(100 - self.nninfo.traning.Error_rate * 100, '.4f') + "% \n" + self.nninfo.traning.Data_set_size.__str__() + " \n" + self.nninfo.traning.Iteration_times.__str__() + " "
            self.traning_log_msg_lab.set_text(tmp_str)
            tmp_str = format(100 - self.nninfo.testing.Error_rate * 100.0, '.4f') + "% \n" + self.nninfo.testing.Data_set_size.__str__() + " "
            self.testing_log_msg_lab.set_text(tmp_str)
        elif pannel == "Testing":
            tmp_str = format(100 - self.nninfo.testing.Error_rate * 100.0, '.4f') + "% \n" + self.nninfo.testing.Data_set_size.__str__() + " "
            self.testing_log_msg_lab.set_text(tmp_str)
        elif pannel == "Training":
            tmp_str = format(100 - self.nninfo.traning.Error_rate * 100, '.4f') + "% \n" + self.nninfo.traning.Data_set_size.__str__() + " \n" + self.nninfo.traning.Iteration_times.__str__() + " "
            self.traning_log_msg_lab.set_text(tmp_str)
    def on_clicked_train(self, widget):

        tmp_struct = self.mlp_structure_ety.get_text()
        self.nnetwork = mlp(structure = [int(x) for x in tmp_struct.split(',')], dimension = len(self.training_set[0][0]), err_rate = self.training_err_rate_sb.get_value(),class_middle = self.dataset.get_class_middle(), learning_rate = self.learning_rate_sb.get_value(), training_times = int(self.training_times_sb.get_value()), best_w = self.find_best)
        self.nninfo.traning.Iteration_times, suc, err = self.nnetwork.training(self.training_set, self.traning_draw_paper, self.on_clicked_draw)
        self.nninfo.traning.Error_rate = len(err[0])/(len(err[0]) + len(suc[0]))
        self.log_refresh(pannel="Training")
        # print("mse", self.nnetwork.get_mse())
        # self.weights = self.nnetwork.get_weights()
        # print(self.weights)
        if len(self.training_set[0][0]) == 2:
            draw_thread = threading.Thread(target=self.draw_area())
            draw_thread.start()
            draw_thread.join()

        draw_thread = threading.Thread(target=self.on_clicked_draw(self.traning_draw_paper, suc, err, self.dataset.get_class_middle()))
        draw_thread.start()
        draw_thread.join()

    def draw_area(self):
        cm = self.dataset.get_class_middle()
        self.ori_paper.resetpaper()
        self.ori_paper.draw_2d_point(self.dataset.get_data(1),cm)
        self.ori_paper.expend_lim()
        density = 100

        # x = np.linspace(0,1,density)
        # y = np.linspace(0,1,density)

        x = np.linspace(-0.02, 1.02,density)
        y = np.linspace(-0.02, 1.02,density)
        x, y = np.meshgrid(x, y)
        area_data = []
        tmp_x = []
        tmp_y = []
        tmp_data = []

        for i in x:
            tmp_x.extend(list(i))

        for i in y:
            tmp_y.extend(list(i))
        for i, j in zip(tmp_x, tmp_y):
            tmp_data.append([i,j])

        z = self.nnetwork.classifier(tmp_data)

        zp = []
        for i in cm:
            if z.count(i) == 0:
                return
        for c in z:
            zp.extend([cm.index(c)])
        z = zp
        # print("x len", len(x))

        tmp = []
        for j in range(len(y)):
            # print(j * len(x), ", ", j * (len(x)) + len(x) - 1)
            tmp.append(z[j * density:j * density + density])
        # print("z", z, ", ", len(tmp), ", ", len(tmp[0]))
        z = np.vstack(tmp)

        cMap = colors.ListedColormap(["r","b","g","c","m","y","k","w"][0:len(cm)])
        self.ori_paper.ax.pcolormesh(x,y,z,cmap=cMap, alpha=0.4)
        self.ori_paper.draw()
    def on_clicked_test(self, widget):
        suc, err = self.nnetwork.testing(self.testing_set)
        self.nninfo.testing.Error_rate = len(err[0])/(len(err[0]) + len(suc[0]))
        draw_thread = threading.Thread(target=self.on_clicked_draw(self.testing_draw_paper, suc, err, self.dataset.get_class_middle()))
        draw_thread.start()
        draw_thread.join()
        self.log_refresh(pannel="Testing")
        # print(self.nninfo.testing.Error_rate)
    def on_clicked_run(self, widget):
        self.on_clicked_train(widget)
        self.on_clicked_test(widget)
    @staticmethod
    def on_clicked_draw(draw_paper, suc, err, class_middle):
        draw_paper.resetpaper()
        draw_paper.draw_2d_point(suc, class_middle)
        draw_paper.draw_2d_point(err, class_middle, 'x')
        draw_paper.expend_lim()
        draw_paper.draw()
    def on_clicked_generate(self, widght):
        if self.dataset.get_data_size() > 100:
            print("tt rate\t", self.traning_testing_rate_sb.get_value())
            self.training_set = self.dataset.get_data(self.traning_testing_rate_sb.get_value() / 100)
            self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
            self.testing_set = self.dataset.get_data(1 - self.traning_testing_rate_sb.get_value() / 100)
            self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])
        else:
            self.training_set = self.dataset.get_data(1)
            self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
            self.testing_set = self.dataset.get_data(1)
            self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])
        self.nninfo.traning.Data_set_size = len(self.training_set[1])
        self.nninfo.testing.Data_set_size = len(self.testing_set[1])

    def on_menu_others(self, widget):
        print("Menu item " + widget.get_name() + " was selected")
    #
    # def on_menu_choices_changed(self, widget, current):
    #     print(current.get_name() + " was selected.")
    #
    # def on_menu_choices_toggled(self, widget):
    #     if widget.get_active():
    #         print(widget.get_name() + " activated")
    #     else:
    #         print(widget.get_name() + " deactivated")
    #
    # def on_button_press_event(self, widget, event):
    #     # Check if right mouse button was preseed
    #     if event.type == Gdk.EventType.BUTTON_PRESS and event.button == 3:
    #         self.popup.popup(None, None, None, None, event.button, event.time)
    #         return True # event has been handled

def main():
    window = nNetwork()
    window.connect("delete-event", Gtk.main_quit)
    window.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()
